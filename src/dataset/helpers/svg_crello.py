"""Original implementation directly parsing crawled data.
"""

import logging
import os
import xml.etree.ElementTree as ET
from itertools import groupby
from typing import List, Tuple, Dict, Any, Optional
from torch import Tensor
import torch
from src.dataset.helpers.util import convert_xywh_to_ltrb
from .cluster_store import FullClusterStore
from einops import rearrange
from itertools import chain
import base64
from PIL import Image, ImageDraw, ImageFont
import io
import math
import requests
import re
from fontTools.ttLib.woff2 import compress

NS = {
    "svg": "http://www.w3.org/2000/svg",
    "xlink": "http://www.w3.org/1999/xlink",
    "xhtml": "http://www.w3.org/1999/xhtml",
}
ET.register_namespace("", NS["svg"])
ET.register_namespace("xlink", NS["xlink"])
ET.register_namespace("html", NS["xhtml"])

logger = logging.getLogger(__name__)

DUMMY_TEXT = "AAAAAAAAAAAAAAAAAA"
CSS_PATH = os.path.join(os.path.dirname(__file__), "crello/fonts.css")


def load_fonts_css(path: str):
    """Load font-family to stylesheet rules mapping.
    Get css from
    """
    import tinycss

    parser = tinycss.make_parser("fonts3")
    stylesheet = parser.parse_stylesheet_file(path)
    faces = [
        {
            decl.name: decl.value.as_css().replace("_old", "")
            for decl in rule.declarations
        }
        for rule in stylesheet.rules
    ]
    return {
        face: list(it) for face, it in groupby(faces, lambda x: x.get("font-family"))
    }


def encode_font_to_base64(font_url):
    # Remove url() if present
    if font_url.startswith("url(") and font_url.endswith(")"):
        font_url = font_url[4:-1].strip("'\"")
    response = requests.get(font_url)
    response.raise_for_status()  # Ensure we raise an error for bad responses
    encoded_string = base64.b64encode(response.content).decode("utf-8")
    return encoded_string


class FontProcessor:
    def __init__(
        self, fonts: Dict[str, Any], font_file_dir: str, use_css: bool = False
    ):
        self._fonts = fonts
        self._font_file_dir = font_file_dir
        self._use_css = use_css

    def _match_font_family(self, font_name: str) -> Optional[str]:
        """Match the given font name to a font family in the CSS."""
        normalized_font_name = font_name.replace(" ", "").lower()

        for css_font_name in self._fonts.keys():
            if css_font_name.replace(" ", "").lower() == normalized_font_name:
                return css_font_name

        return None

    def generate_font_style(self, font_name: str, font_path: str) -> Optional[str]:
        if self._use_css:
            return self._generate_css_font_style(font_name)
        else:
            return self._generate_base64_font_style(font_name, font_path)

    def _generate_css_font_style(self, font_name: str) -> Optional[str]:
        matched_font_family = self._match_font_family(font_name)
        if not matched_font_family:
            return None

        font_styles = self._fonts.get(matched_font_family)
        if not font_styles:
            return None

        css_rules = []
        for style in font_styles:
            css_rule = "@font-face {\n"
            for key, value in style.items():
                css_rule += f"    {key}: {value};\n"
            css_rule += "}"
            css_rules.append(css_rule)

        return "\n".join(css_rules)

    def _generate_base64_font_style(
        self, font_name: str, font_path: str
    ) -> Optional[str]:
        try:
            encoded_font = self._encode_font_to_base64(font_path)
        except Exception as e:
            print(f"Error encoding font {font_name}: {str(e)}")
            return None

        return f"""
        @font-face {{
            font-family: '{font_name}';
            font-style: normal;
            font-weight: 400;
            font-display: swap; 
            src: url(data:application/octet-stream;base64,{encoded_font}) format('woff');
        }}
        """

    @staticmethod
    def _encode_font_to_base64(font_path: str) -> str:
        with open(font_path, "rb") as font_file:
            return base64.b64encode(font_file.read()).decode("utf-8")

    def find_font_file(self, font_name: str) -> Optional[str]:
        for file in os.listdir(self._font_file_dir):
            if re.search(font_name, file, re.IGNORECASE):
                return os.path.join(self._font_file_dir, file)
        return None


class SVGBuilder(object):
    def __init__(
        self,
        serialize_cluster_store: FullClusterStore,
        _special_token_name2id: int,
        cat_mapping: Any,
        canvas_width: int = None,
        canvas_height: int = None,
        canvas_background_color: Tuple[int, int, int] = None,
        opacity: int = 0.5,
        use_css: bool = False,
        render_partial: bool = False,
    ):
        self._serialize_cluster_store = serialize_cluster_store
        self._canvas_width = canvas_width or 256
        self._canvas_height = canvas_height or 256
        self._canvas_background_color = canvas_background_color or (255, 255, 255)
        self._opacity = opacity
        self._special_token_name2id = _special_token_name2id
        self._cat_mapping = cat_mapping
        self._fonts = self._get_font_family(load_fonts_css(CSS_PATH))
        self._font_file_dir = os.path.join(
            serialize_cluster_store._config.data_root.split("filter")[0],
            "font/raw/woff",
        )
        self._font_processor = FontProcessor(self._fonts, self._font_file_dir, use_css)
        self._use_css = use_css
        self._render_partial = render_partial
        self._render_images = True
        image_prefix = os.path.join(self._serialize_cluster_store._config.data_root, "images/")
        if not os.path.exists(image_prefix):
            self._render_images = False
            logger.warning(f"Images for retrieval not found: {image_prefix}")
            logger.warning("Please check if the images are downloaded.")
            logger.warning("Images will not be rendered.")

    def __call__(self, elements_: List[Tensor], canvas_attr: Tensor):
        # Set canvas properties
        canvas_width = (
            canvas_attr[1].item() if canvas_attr[1].item() > 0 else self._canvas_width
        )
        canvas_height = (
            canvas_attr[2].item() if canvas_attr[2].item() > 0 else self._canvas_height
        )
        # canvas_background_color = self._rgb2hex(self._serialize_cluster_store.get_bc_RGB_from_tokens(canvas_attr[3])[0])
        canvas_background_color = self._serialize_cluster_store.get_bc_RGB_from_tokens(
            canvas_attr[3]
        )[0]
        doc_size = {"width": canvas_width, "height": canvas_height}
        layouts = []
        # Create the root SVG element
        root = ET.Element(
            ET.QName(NS["svg"], "svg"),
            {
                "width": str(canvas_width),
                "height": str(canvas_height),
                "style": f"background-color:rgb{canvas_background_color};",
                "viewBox": f"0 0 {canvas_width} {canvas_height}",
                "preserveAspectRatio": "none",
            },
        )
        # compute the render order of elements
        # elements = self._rule_based_sort(elements_)
        if elements_.shape[0] == 0:
            elements = torch.empty(0, 9, dtype=torch.int64)
        else:
            elements = torch.stack(sorted(elements_, key=lambda x: x[6]))
        # Render each element
        for element in elements:
            # Determine if the element is text or a non-text element
            element_type = self._determine_element_type(element)
            if element_type not in ['text','non-text']:
                continue
            if self._render_partial:
                bbox = self._get_bbox_from_element(element)
                # Store [c, x, y, w, h] where c is the category token (element[1]) for computing saliency metrics
                layouts.append([element[1].item(), bbox[0].item(), bbox[1].item(), bbox[2].item(), bbox[3].item()])
            if element_type == "text":
                # Skip text elements if render_text is False
                if self._render_partial:
                    continue
                self._render_text_element(
                    root, element, doc_size, canvas_width, canvas_height
                )
            elif element_type == "non-text":
                if not self._render_images:
                    continue
                self._render_non_text_element(
                    root, element, canvas_width, canvas_height
                )

        style = ET.SubElement(root, "{%s}style" % NS["svg"])
        self._fill_stylesheet(root, style)

        # Convert the tree to a string
        final_svg = ET.tostring(root, encoding="utf-8").decode("utf-8")
        # cairosvg.svg2png(bytestring=final_svg.encode('utf-8'),write_to='sample.png')
        # final_png = cairosvg.svg2png(bytestring=final_svg.encode("utf-8"))
        # final_image = Image.open(io.BytesIO(final_png))
        # patches = [self._crop_patch(final_image, element) for element in elements_]
        if self._render_partial:
            return final_svg,layouts
        else:
            return final_svg,None

    def _render_text_element(
        self,
        parent,
        element: Tensor,
        doc_size: Dict[str, int],
        canvas_width: int,
        canvas_height: int,
    ):
        # [tex] [cat] [x] [y] [w] [h] [z] [ff] [fs] [fc]
        bbox = self._get_bbox_from_element(element)
        font_family = self._serialize_cluster_store.get_ff_ttf_from_token(
            element[7].numpy()
        )
        font_size = self._serialize_cluster_store.get_fs_pt_from_tokens(
            element[8].numpy()
        ).item()
        font_size__ = font_size * canvas_height
        font_color = self._serialize_cluster_store.get_fc_RGB_from_tokens(
            element[9].numpy()
        )[0]
        
        margin = bbox[3] * canvas_width * 0.1  # To avoid unexpected clipping.
        # text y position is the baseline of the text
        lines = int(bbox[-1] / font_size)
        y_center = bbox[1] * canvas_height + bbox[-1] * 0.5
        # if lines <= 1:
        #     y = y_center + font_size__ * 0.5
        # else:
        #     y = y_center + font_size__ * 0.5 - (lines - 1) * font_size__ * 0.5
        if lines <= 1:
            lines = 1
        margin_ = (bbox[3] * canvas_height - font_size__ * lines) / 2
        # Create the text element with the obtained properties
        container = ET.SubElement(
            parent,
            ET.QName(NS["svg"], "svg"),
            {
                "x": "%g" % (bbox[0] * canvas_width or 0),
                # "y": "%g" % ((bbox[1] * canvas_height + font_size__ or 0)),# - margin),
                "y": "%g" % ((bbox[1] * canvas_height or 0) - margin_),
                "width": "%g" % (bbox[2] * canvas_width),
                "overflow": "visible",
            },
        )
        text_align = getattr(element, "textAlign", "center")
        line_height = getattr(element, "lineHeight", 1.0)
        capitalize = getattr(element, "capitalize", False)
        underline = getattr(element, "underline", False)
        letter_spacing = getattr(element, "letterSpacing", 0.0)
        # Create the text element with the obtained properties
        text_element = ET.SubElement(
            container,
            "{%s}text" % NS["svg"],
            {
                "font-family": font_family,
                "font-size": "%g" % font_size__,
                "letter-spacing": "%g" % letter_spacing,
            },
        )
        if underline:
            text_element.set("text-decoration", "underline")
        x = {"left": "0", "center": "50%", "right": "100%"}[text_align]
        anchor = {"left": "start", "center": "middle", "right": "end"}[text_align]
        line_height = line_height * font_size__
        line_tspan = ET.SubElement(
            text_element,
            "{%s}tspan" % NS["svg"],
            {
                "dy": "%g" % line_height,
                "x": x,
                "text-anchor": anchor,
                "dominant-baseline": "central",
                "fill": f"{self._rgb2hex(font_color)}",
            },
        )

        if capitalize:
            # Capitalize at the leaf span for Safari compatibility.
            line_tspan.set("style", "text-transform: uppercase;")

        text_count = max(math.ceil(bbox[2] / font_size * 0.5), 3)
        line_tspan.text = "A" * text_count

        lines = int(bbox[-1] / font_size)
        for i in range(1, lines):
            line_tspan_ = ET.SubElement(
                text_element,
                "{%s}tspan" % NS["svg"],
                {
                    "dy": "%g" % line_height,
                    "x": x,
                    "text-anchor": anchor,
                    "dominant-baseline": "central",
                    "fill": f"{self._rgb2hex(font_color)}",
                },
            )
            if capitalize:
                # Capitalize at the leaf span for Safari compatibility.
                line_tspan_.set("style", "text-transform: uppercase;")
            line_tspan_.text = "A" * text_count

        return container

    def _render_non_text_element(
        self, parent, element: Tensor, canvas_width: int, canvas_height: int
    ):
        # [ntx] [cat] [x] [y] [w] [h] [z] [img] [pad] [pad]
        bbox = self._get_bbox_from_element(element)
        image_path = self._serialize_cluster_store.get_patch_from_token(
            element[7].item(), element[1].item()
        )
        image_url = self._image_to_base64(image_path)
        # Create an image element and place it in the bbox
        ET.SubElement(
            parent,
            ET.QName(NS["svg"], "image"),
            {
                "x": str(bbox[0] * canvas_width),
                "y": str(bbox[1] * canvas_height),
                "width": str(bbox[2] * canvas_width),
                "height": str(bbox[3] * canvas_height),
                ET.QName(NS["xlink"], "href"): image_url,
                "preserveAspectRatio": "none",
            },
        )

    def _make_rect(self, parent, element, fill):
        bbox = self._get_bbox_from_element(element)
        return ET.SubElement(
            parent,
            ET.QName(NS["svg"], "rect"),
            {
                "x": str(bbox[0]),
                "y": str(bbox[1]),
                "width": str(bbox[2]),
                "height": str(bbox[3]),
                "fill": str(fill),
            },
        )

    def _determine_element_type(self, element: Tensor):
        if element[0] == self._special_token_name2id["tex"]:
            return "text"
        elif element[0] == self._special_token_name2id["ntx"]:
            return "non-text"
        elif element[0] == self._special_token_name2id["pad"]:
            return "pad"
        else:
            raise ValueError(f"Unknown element type: {element[0]}")

    def _filter_font_style(self, font_family_path):
        """Filter the font dictionary to return the regular variant of the given font family.
        If 'regular' is not available, it returns the 'normal' style and '400' weight variant.
        """
        font_family_name = font_family_path.split("/")[-1].split(".")[0]
        variants = self._fonts.get(font_family_name, [])

        # Filter for 'Regular' or 'normal' and '400' weight.
        regular_variant = [
            variant
            for variant in variants
            if variant.get("font-style", "normal") == "normal"
            and variant.get("font-weight", "400") == "400"
        ]

        # If there's a 'Regular' variant, return it, otherwise return the normal style, 400 weight variant.
        if regular_variant:
            return regular_variant[0]
        else:
            return None

    def _get_font_family(self, fonts: Dict[str, Any]):
        styles_to_remove = ["BoldItalic", "Italic", "Bold"]
        new_fonts = {}
        for key, value in fonts.items():
            if key.endswith("Regular"):
                new_fonts[key.replace(" Regular", "")] = value
            elif not any(key.endswith(style) for style in styles_to_remove):
                new_fonts[key] = value
        return new_fonts

    def _image_to_base64(self, image, fmt="png") -> str:
        if type(image) == str:  # image_path
            with open(image, "rb") as image_file:
                byte_data = image_file.read()
        else:
            output_buffer = io.BytesIO()
            try:
                image.save(output_buffer, format=fmt)
            except:
                # TODO: error bbox / canvas
                return None
            byte_data = output_buffer.getvalue()
        base64_str = base64.b64encode(byte_data).decode("utf-8")
        return f"data:image/{fmt};base64," + base64_str

    def _fill_stylesheet(self, root, style):
        font_families = {
            text.get("font-family")
            for text in root.iter("{%s}text" % NS["svg"])
            if text.get("font-family") is not None
        }

        styles = []
        for family in font_families:
            if self._use_css:
                css_style = self._font_processor.generate_font_style(family, None)
            else:
                font_file = self._font_processor.find_font_file(family)
                if font_file:
                    css_style = self._font_processor.generate_font_style(
                        family, font_file
                    )
                else:
                    print(f"Font file not found for {family}")
                    css_style = None
            if css_style:
                styles.append(css_style.strip())

        style.text = "\n".join(styles)

    def _rgb2hex(self, rgb: Tuple[int, int, int]) -> str:
        return "#%02x%02x%02x" % rgb

    def _rule_based_sort(
        self, elements: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sort elements by rule-based order."""
        if elements.size(0) == 0:
            return elements
        text_elements = elements[elements[:, 0] == self._tex_token]
        non_text_elements = elements[elements[:, 0] != self._tex_token]

        # Manually define the order of non-text elements
        category_order = {
            "coloredBackground": 0,
            "maskElement": 1,
            "imageElement": 2,
            "svgElement": 3,
        }
        non_text_elements = sorted(
            non_text_elements,
            key=lambda x: (
                category_order[self._cat_mapping(x[1].item())],
                -self._get_size_element(x),
            ),
        )
        if non_text_elements:
            non_text_elements = torch.stack(non_text_elements, dim=0).to(torch.int64)
        else:
            non_text_elements = torch.empty(0, 9, dtype=torch.int64)
        # Concatenate the text and non-text elements
        sorted_elements = torch.cat([non_text_elements, text_elements], dim=0)

        # # Sort the elements by their y-coordinates
        # _, indices = torch.sort(
        #     torch.arange(0, elements.size(0)), key=lambda i: sorted_elements[i]
        # )

        return sorted_elements

    def _get_size_element(self, element: Tensor):
        bbox = self._get_bbox_from_element(element)
        return bbox[2] * bbox[3]

    def _crop_patch(self, image: Image.Image, element: Tensor) -> Image.Image:
        bbox_ = self._get_bbox_from_element(element)
        canvas_width = image.size[0]
        canvas_height = image.size[1]
        bbox = [
            bbox_[0] * canvas_width,
            bbox_[1] * canvas_height,
            bbox_[2] * canvas_width,
            bbox_[3] * canvas_height,
        ]
        return image.crop((bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]))

    def _get_bbox_from_element(self, element: Tensor) -> Tensor:
        pos_token = rearrange(element[2:6], "A -> 1 A")
        bbox = rearrange(
            self._serialize_cluster_store.get_bbox_from_tokens(pos_token), "1 A -> A"
        )
        return bbox

    def _draw_dummy_text(
        self, font_family, font_size, font_color, width, height, dummy_text
    ):
        try:
            font = ImageFont.truetype(font_family, font_size)
            x1, y1, x2, y2 = font.getbbox(dummy_text)
        except:
            # TODO: Fix the lost font
            font_family = font_family.split("raw")[0] + "raw/Montserrat.ttf"
            font = ImageFont.truetype(font_family, font_size)
            x1, y1, x2, y2 = font.getbbox(dummy_text)
        while x2 - x1 > width:
            dummy_text = dummy_text[:-1]
            x1, y1, x2, y2 = font.getbbox(dummy_text)
        if dummy_text == "":
            dummy_text = "A"
        w = x2 - x1
        image = Image.new(mode="RGBA", size=(width, height))
        draw_table = ImageDraw.Draw(im=image)
        draw_table.text(
            xy=((width - w) / 2, 5),
            text=dummy_text,
            fill=f"{self._rgb2hex(font_color)}",
            align="center",
            font=font,
        )

        return image
