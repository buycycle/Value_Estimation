# Local and Development

setup:
	conda create  --name price --file requirements.txt python=3.9

activate:
	conda activate .env/price

juypter:
	@cd notebook; PYTHONPATH=".." jupyter notebook notebook.ipynb

test:
	# run integration test
	pytest -v tests/test_fastapi.py

test_unit:
	# run test cases in tests directory, optional with -s for debugging
	pytest -v --ignore=tests/test_fixtures.py --ignore=tests/test_fastapi.py tests/

lint:
	pylint --disable=R,C,W1203,W1202 src/price.py
	pylint --disable=R,C,W1203,W1202 model/app.py



	# check style with flake8
	mypy .
	flake8 .

format:
	# Check Python formatting
	black --line-length 130 .



build-docker:
	## Run Docker locally

	docker compose build

	docker compose up app


all: install lint test

# Deployment

APP_NAME := price
VALUES_FILE = values.yaml
CLONE_DIR := /tmp/$(APP_NAME)/$(ENV)
REPO_NAME := buycycle-helm
TARGET_DIR := $(CLONE_DIR)/$(REPO_NAME)
GIT_USERNAME := Jenkins
GIT_EMAIL := jenkins@buycycle.de
BRANCH := main
REPO_URL := git@gitlab.com:buycycle

configure-git:
	git config --global user.name $(GIT_USERNAME)
	git config --global user.email $(GIT_EMAIL)

clone: clear configure-git
	mkdir -p $(CLONE_DIR) && cd $(CLONE_DIR) && git clone $(REPO_URL)/$(REPO_NAME).git

clear:
	test -d $(TARGET_DIR) && rm -R -f $(TARGET_DIR) || true

modify:
	yq -i ".image.tag = \"$(IMAGE_TAG)\"" $(TARGET_DIR)/$(ENV)/$(APP_NAME)/$(VALUES_FILE)

push: clone modify
	cd $(TARGET_DIR) && git add $(ENV)/$(APP_NAME)/$(VALUES_FILE) && git commit -m "updated during build $(APP_NAME) $(IMAGE_TAG)" && git pull --rebase && git push -u origin $(BRANCH)
