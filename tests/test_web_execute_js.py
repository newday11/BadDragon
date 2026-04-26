from __future__ import annotations

import unittest

from app.tools import web_execute_js


class WebExecuteJsTests(unittest.TestCase):
    def test_parse_window_open_with_target(self) -> None:
        result = web_execute_js.run(
            {"script": "window.open('https://www.baidu.com', '_blank');"}
        )
        output = result.get("output", {})
        self.assertEqual(output.get("action"), "open_new_tab")
        self.assertEqual(output.get("url"), "https://www.baidu.com")

    def test_parse_location_href(self) -> None:
        result = web_execute_js.run(
            {"script": 'window.location.href = "https://www.baidu.com";'}
        )
        output = result.get("output", {})
        self.assertEqual(output.get("action"), "navigate")
        self.assertEqual(output.get("url"), "https://www.baidu.com")


if __name__ == "__main__":
    unittest.main()
