"""Lets `python -m sales_poc.datagen` run the generate-data command."""

import sys

from sales_poc.datagen.cli import main

sys.exit(main())
