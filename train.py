"""Train or evaluate the main EGRC model."""

import json

from egrc.config import parse_args


if __name__ == "__main__":
    args = parse_args()
    if args.dry_run:
        print(json.dumps(vars(args), indent=2))
    else:
        from egrc.train import main

        main(args)
