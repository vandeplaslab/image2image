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

# examples:
# make watch ~/Desktop/Untitled.png
# make watch -- -w animation  # -- is required for passing flags to autoims

watch:
	@echo "running: autoims $(WATCH_ARGS)"
	@echo "Save any file to restart autoims\nCtrl-C to stop..\n" && \
		watchmedo auto-restart -R \
			--ignore-patterns="*.pyc*" -D \
			--signal SIGKILL \
			--directory src \
			--interval 10 \
            --verbose \
			autoims -- $(WATCH_ARGS) || \
		echo "please run 'pip install watchdog[watchmedo]'"


.PHONY: untrack
untrack:
	git rm -r --cached .
	git add .
	git commit -m ".gitignore fix"
