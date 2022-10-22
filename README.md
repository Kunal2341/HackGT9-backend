



# HackGT9-backend

```
conda create -n hackgt-9 python=3.9 
conda activate hackgt-9 
pip install -r requirements.txt 
```
# Running the server

uvicorn app:app --reload

# Testing Server


```
client = TestClient(app)
client.post("/update/'./test_img.jpeg'")
print(client.get("/mapping"))
```