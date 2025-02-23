**Facelock - Face Recognition App**

A face recognition-based access control system using OpenCV and Flask. This project captures face images, processes them into embeddings, and enables authentication via a web interface.

**🚀 Features**

✅ Face Image Collection – Capture and store images of known faces
✅ Face Embedding Training – Convert images into trained embeddings for recognition
✅ Web-Based Authentication – Authenticate users through a Flask web interface
✅ Model Storage – Saves trained embeddings in a .pkl file for fast recognition

**🛠 Installation & Setup**

Follow these steps to set up and run the project on your system.

1️⃣ Clone the Repository

	git clone https://github.com/yourusername/FACEID.git
	cd FACEID

2️⃣ Create a Virtual Environment

	python3 -m venv venv
	source venv/bin/activate   # macOS/Linux  
	venv\Scripts\activate      # Windows

3️⃣ Install Dependencies

	pip install -r requirements.txt

**🔧 How to Use**

The project consists of three main steps: Collecting images, Training face embeddings, and Running the web application.

📷 Step 1: Collect Face Images
	Run the script to collect images for each user:
 	•	This will capture multiple face images per user and store them in the dataset/ folder.
	•	Ensure you provide a unique user ID while collecting images.
	
 	python script.py
	

🧠 Step 2: Train Face Embeddings
	Convert images into a trained model for recognition:
	•	This script processes face images, extracts embeddings, and saves them in embeddings.pkl.
	•	The trained embeddings allow the system to recognize users based on their stored images.

  	python embeddingtrain.py

🌍 Step 3: Run the Web App
	Start the Flask web application:
 	•	Opens a web interface for face authentication.
	•	Matches live webcam input against stored face embeddings.

 	 
 	python app.py

**📂 Folder Structure**

FACEID/
	│── dataset/              	# Stores images of known faces
	│── templates/            	# HTML templates for the web interface
	│── script.py             	# Collects face images
	│── embeddingtrain.py     	# Converts images into trained embeddings
	│── app.py                	# Runs Flask application
	│── requirements.txt      	# Project dependencies
	│── README.md             	# Documentation

**📝 License**
MIT License © 2025 Sanjay Chandrasekar
