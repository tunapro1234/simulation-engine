"""README stays in step with the hackathon recipes: every examples/recipes/*.py and docs/recipes/*.md is linked from
the "Hackathon reçeteleri" section, the English summary points English readers at them, and every relative link
in the README resolves to a file or directory in the repo.

    python -m unittest tests.test_readme_recipes
"""
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
README = (ROOT / "README.md").read_text(encoding="utf-8")
LINK = re.compile(r"\]\(([^)\s]+)\)")


def _section(heading: str) -> str:
    m = re.search(r"^## " + re.escape(heading) + r"\s*$(.*?)(?=^## |\Z)", README, re.M | re.S)
    return m.group(1) if m else ""


class ReadmeRecipesTest(unittest.TestCase):
    def test_every_recipe_and_guide_linked(self):
        sec = _section("Hackathon reçeteleri")
        self.assertTrue(sec, "README has no '## Hackathon reçeteleri' section")
        links = set(LINK.findall(sec))
        scripts = sorted((ROOT / "examples" / "recipes").glob("*.py"))
        guides = sorted((ROOT / "docs" / "recipes").glob("*.md"))
        self.assertTrue(scripts and guides)
        for p in scripts + guides:
            self.assertIn(p.relative_to(ROOT).as_posix(), links, f"{p.name} missing from the recipes table")

    def test_english_summary_mentions_recipes(self):
        summary = README.split("\n---", 1)[0]
        self.assertIn("examples/recipes/", summary)
        self.assertIn("docs/recipes/", summary)

    def test_relative_links_resolve(self):
        bad = []
        for target in LINK.findall(README):
            if re.match(r"[a-z]+:", target) or target.startswith("#"):
                continue
            path = target.split("#", 1)[0]
            if path and not (ROOT / path).exists():
                bad.append(target)
        self.assertEqual(bad, [])


if __name__ == "__main__":
    unittest.main()
