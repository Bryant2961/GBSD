"""Execute official runtime and parameter analysis derived from main runs."""

import sys
from generate_all import main

if __name__ == "__main__":
    if "--stage" not in sys.argv and "--all" not in sys.argv:
        sys.argv[1:1] = ["--stage", "runtime"]
    raise SystemExit(main())
