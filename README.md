## 💡 Inspiration

As the fast-paced nature of our lives continues to be a stressor for many, people are now striving to discover newer ways to stay in touch with their creative side. Keeping the need for an artistic outlook in mind, we wish to connect the two outlooks that best express people's creativity: art and music. We believe that a fulfilling means of exploring their imaginations would allow people to be more mindful of their surroundings and find a way to rejuvenate.

To provide another means of expression, we created Board Band, a solution that combines art and music while giving users unlimited freedom to express their creativity. 

# Repository structure

```shell
├── apps
│   ├── client
│   └── smart-contracts
├── packages
│   └── eslint-config-custom
├── .github
├── .eslintrc.js
├── gitignore
├── package.json
├── README.md
├── turbo.json
└── yarn.lock
```

## 💻 What it does
- Combines art and music for entertainment purposes
- Allows users to play musical instruments and compose songs using arbitrary drawings
- Facilitates the composition of music using the user's touch on a surface
- Serves as a means of entertainment that allows users to rejuvenate and express themselves creatively


## ⚙️ How we built it
- Captures Hand Pose from Java Script every x frames
- Captures a baseline capturing each shape drawn on whiteboard
- Generates a mask of the shapes drawn based on the color of the marker
- Seperates and identifies each shape
- Correlate each shape to a musical instrument based on the following [`Vertices` - `Size` - `Complexity through center`]
- Get depth of hand from hand pose front end function and determine when hand was clicked
- Play sound

### Youtube Sound *(_Still in progress_)
- Input Youtube link
- Download song 
- Split song into each note (every secound approx)
- Run algorithm to determine which note is played
- Represent same note to a image on whiteboard to be replayed


## 🧠 Challenges we ran into
- Tuning the right mask threshold to get the correct bounding boxes of shapes
- Developing bounding boxes in different environments
- Mapping the coordinates of  complex shapes
- Detecting user touch by calculating the depth of the user's finger




## 🏅 Accomplishments that we're proud of

- Implementing a working and functioning prototype of our idea
- Designing and developing a minimalist and clean user interface through a new UI library and reusable components with an integrated design
- Create a program that accurately recognizes user touch and acts with little lag


## 📖 What we learned
- Full stack development along with backend python integration with front end JavaScript


## 🚀 What's next for Board Band

- Implementing more sounds and sheet music to allow users to test their skills 
- Adding sheet music to allow users to learn new music through shapes


# HackGT9-backend

```
conda create -n hackgt-9 python=3.9 
conda activate hackgt-9 
pip install -r requirements.txt 
```
# Running the server

```
uvicorn app:app --reload
```
# Testing Server


```
client = TestClient(app)
client.post("/update/'./test_img.jpeg'")
print(client.get("/mapping"))
```