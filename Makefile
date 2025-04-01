.PHONY: pre watch dist settings-schema untrack

# note: much faster to run mypy as daemon,
# dmypy run -- ...
# https://mypy.readthedocs.io/en/stable/mypy_daemon.html
typecheck:
	tox -e mypy

check-manifest:
	pip install -U check-manifest
	check-manifest

dist: check-manifest
	pip install -U build
	python -m build

pre:
	pre-commit run -a

# If the first argument is "watch"...
ifeq (watch,$(firstword $(MAKECMDGOALS)))
  # use the rest as arguments for "watch"
  WATCH_ARGS := $(wordlist 2,$(words $(MAKECMDGOALS)),$(MAKECMDGOALS))
  # ...and turn them into do-nothing targets
  $(eval $(WATCH_ARGS):;@:)
endif

.PHONY: untrack
untrack:
	git rm -r --cached .
	git add .
	git commit -m ".gitignore fix"
