# HackGT9-backend

# Create and activate conda environment

conda create -n hackgt-9 python=3.9 <br/>
conda activate hackgt-9 <br/>
pip install -r requirements.txt <br/>

# Running the server

uvicorn app:app --reload

# Testing Server

client = TestClient(app) <br/>
client.post("/update/'./test_img.jpeg'") <br/>
print(client.get("/mapping")) <br/>
