"""`python -m simengine ...` = the `simengine` command (works without installing: PYTHONPATH=.)."""
from .cli import main

raise SystemExit(main())
