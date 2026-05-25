"""
tests/test_crop_from_cvat.py
────────────────────────────
Unit tests for CVAT XML parsing and crop logic.
Uses a minimal in-memory XML string — no file I/O needed.
"""

import io
import pytest
from unittest.mock import patch, MagicMock
from src.data_prep.crop_from_cvat import parse_cvat_xml


MINIMAL_XML = b"""<?xml version="1.0" encoding="utf-8"?>
<annotations>
  <version>1.1</version>
  <image id="0" name="img001.jpg" width="800" height="600">
    <box label="barcode" xtl="10.0" ytl="20.0" xbr="200.0" ybr="80.0"/>
    <box label="product_id" xtl="220.0" ytl="30.0" xbr="450.0" ybr="70.0"/>
    <box label="blood_group" xtl="460.0" ytl="30.0" xbr="600.0" ybr="70.0"/>
    <box label="expiry_date" xtl="10.0" ytl="100.0" xbr="250.0" ybr="140.0"/>
  </image>
  <image id="1" name="img002.jpg" width="800" height="600">
    <box label="product_id" xtl="50.0" ytl="50.0" xbr="300.0" ybr="90.0"/>
  </image>
</annotations>
"""


class TestParseCvatXml:

    def test_returns_correct_count(self, tmp_path):
        xml_path = tmp_path / "annotations.xml"
        xml_path.write_bytes(MINIMAL_XML)
        records = parse_cvat_xml(str(xml_path))
        assert len(records) == 5  # 4 from img001 + 1 from img002

    def test_label_names(self, tmp_path):
        xml_path = tmp_path / "annotations.xml"
        xml_path.write_bytes(MINIMAL_XML)
        records = parse_cvat_xml(str(xml_path))
        labels = {r["label"] for r in records}
        assert "barcode_region" in labels
        assert "donation_id" in labels
        assert "blood_group" in labels
        assert "expiry_date" in labels

    def test_box_coordinates_are_ints(self, tmp_path):
        xml_path = tmp_path / "annotations.xml"
        xml_path.write_bytes(MINIMAL_XML)
        records = parse_cvat_xml(str(xml_path))
        for r in records:
            x1, y1, x2, y2 = r["box_xyxy"]
            assert isinstance(x1, int)
            assert x2 > x1
            assert y2 > y1

    def test_image_names_preserved(self, tmp_path):
        xml_path = tmp_path / "annotations.xml"
        xml_path.write_bytes(MINIMAL_XML)
        records = parse_cvat_xml(str(xml_path))
        img_names = {r["image_name"] for r in records}
        assert "img001.jpg" in img_names
        assert "img002.jpg" in img_names

    def test_img002_has_only_donation_id(self, tmp_path):
        xml_path = tmp_path / "annotations.xml"
        xml_path.write_bytes(MINIMAL_XML)
        records = parse_cvat_xml(str(xml_path))
        img002_records = [r for r in records if r["image_name"] == "img002.jpg"]
        assert len(img002_records) == 1
        assert img002_records[0]["label"] == "donation_id"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
