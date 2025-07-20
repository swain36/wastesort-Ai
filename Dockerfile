# Use official Python 3.9 base image
FROM python:3.9-slim

# Set work directory
WORKDIR /app

# Copy your code into container
COPY . .

# Install dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Run the app
CMD ["streamlit", "run", "app.py", "--server.port=10000", "--server.address=0.0.0.0"]
