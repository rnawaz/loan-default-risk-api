FROM python:3.9-slim

# Set working directory
WORKDIR /code

# Copy requirements first
COPY requirements.txt /code/

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project
COPY . /code/

# Expose port for HuggingFace Spaces
EXPOSE 7860

# Run the FastAPI app
CMD ["python", "app.py"]
