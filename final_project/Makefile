test:
	python -m pytest tests/test_source_code.py

quality_checks:
	isort .
	black .
	pylint --recursive=y . ; echo $$?

setup:
	# Pipenv
	pip install --upgrade pip
	pip install pipenv==2024.0.1
	pipenv install --dev
	pipenv install backports.tarfile #For compatibility reasons

	# Pre-commit hooks
	pre-commit install

build_docker_container: quality_checks test
	docker build -t bank-churn-prediction-service:v1 -f deployment/Dockerfile .

run_docker_container: build_docker_container
	docker run -it --rm -p 9696:9696 bank-churn-prediction-service:v1

prediction_test:
	@if [ $$(docker ps -q -f ancestor=bank-churn-prediction-service:v1) ]; then \
		python -m pytest tests/test_predictions.py; \
	else \
		echo "The Docker container is not running. Please start the server first by running makefile run_docker_container."; \
		exit 1; \
	fi
