[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/EppqwQTz)

# Setup

1. Clone the repo
2. `cd` into the root folder
3. Setup a virtual env
4. `pip install -r requirements.txt`

# Gathering Training Data

> ⚠️ The DIPPID android app sometimes becomes unresponsive after the phone's screen goes to sleep and has to be restarted. If the data gathering process does not react to button presses it is not an issue with this program but with the DIPPID android app!

```sh
python gather_data.py --sets 5 --duration 10 --activitiy "rowing" --prefix "name"
```
> 💡 The process captures data rows at a sampling rate of *200 Hz* by default. You can adjust the sampling rate with the `--sampling-rate` flag

To use this program connect a DIPPID client to the **IP** and **port** that are printed when the process starts.  
*If you want the DIPPID server to run on a different port you can pass it with the `--port` CLI param.*  

This will start the data gathering process for a total of `5` sets that lasts `10` seconds each.  
The resulting sets will be saved to `.csv` files in the data folder with the name schema `<prefix>-<activity>-<set>.csv`.  

Before each set the user is prompted to press `button_1` on the DIPPID device to start capturing for this set, after a short delay (configurable via the `--delay` CLI param)  
capturing for the current set starts. It is recommended to already start the motion during the countdown as to not introduce any weird data inconsistencies.  

When the program is done capturing the current set it will keep prompting the user for the next set until there are no more sets left.  
By pressing `button_2` when prompted to start the next set, users can redo the last set if there were any physical issues during recording.

### Resampling

```sh
python resample.py
```

To get the data sheets at a rate of **100 Hz** as described in [Task 1](./assignment03.pdf) you can either adjust the sampling rate directly or use the modified [resample script](./resample.py).  
This script will resample all `.csv` files in the `data` folder to **100 Hz**. (The script will interpolate missing data points if the existing frequency is below 100 Hz)

# Activity Recognition

> 💡 Activity recognition *can* be started as a console application seperately from the game using `python activity_recognizer --window-seconds 1 --sample-rate 60` (Parameters optional)  
> 💡 If you want to run both at the same time you need to change the `--port` for one of them

## Usage

```sh
python fitness_trainer.py --session-file "sessions/balanced.json"
```

At the first start of the application the model is trained using the training data in [the data folder](./data/).  
It uses all `.csv` files in the folder. All files **must** follow this naming scheme `<name>-<activity>-<setnumber>.csv` or training will fail (the \<activity> is used as `y` for the classifier).  
Subsequent launches of the application will load the model from disk instead of retraining. The first launch can take a while due to cross-validation of parameters to find the best model.  

The fitness trainer application was designed with training set configurations in json format.  
The default session file [balanced.json](./sessions/balanced.json) includes a full sample training session with all activites that the model has been trained on.  

```json
// A session config must follow this format
{
    "stages": [
        {
            "name": "Example Stage Name",
            "activities": [
                {
                    "name": "rowing", // Must be one of the available activities and be unique within the stage
                    "duration": 30, // Duration in seconds for this activity
                }
            ]
        }
    ]
}
```

Once the game has started connect to the DIPPID server (ip and port are printed in the console) and follow the on-screen instructions to complete the training session.  
The overview on the bottom left shows a list of stages and a preview of activities included in them.  
On the bottom right, the activity meter shows if the user is currently doing anything or if they are idle.
The top half shows all activities in the current stage with a progress meter and the required duration, activities progress when the ML classifier predicts them and the activity meter threshold is met.  

### Known Issues

**DIPPID Client Freeze**

On my android device the DIPPID app freezes a few seconds after the screen goes to sleep. If the activity meter suddenly drops or is inconsistent with your movement, check if the DIPPID android app is frozen. If this happens you can simply restart the android app and the `Fitness Trainer` will continue where you left off.

**Idle Detection**

    Context:
    Since the model constantly predicts the activity for the given input it always outputs an activity.  
    When the user e.g. leaves the sensor device on a table, the model still outputs one of the 4 activities. In order to avoid this I implemented a very basic idle check that checks the standard deviation of  
    the accelerometer across a specific window + a hysteresis that only enables/disables idle state after the device has been in the respective state for a set duration.

    I tried different metrics like frequency thresholding or checking for correlation between the parameters however nothing worked as well as simple std across the window.  
    On top of that I had to set the threshold really low because of slow activities like lifting or rowing.  

    I even use probabilities however those probabilities are often not reliable enough to properly detect if the model matched none of the labels. (Maybe I overfitted ¯\_(ツ)_/¯)

Ultimately this results in random activities gaining progress when the user just wildly flings their phone around. (When an activity is actually physically executed correctly it always detected the correct activity during my testing). When the sensor device is laying flat or in a normally swinging/idle hand then idle detection works as well, just random movements are not filtered correctly (This could probably be fixed by using a different classifier or maybe adding "idle" as an activity so the SVC can detect it instead of manually coding idle detection)

**Application Launch Duration**

When the app launches it loads the images for all activities. Apparently this is a blocking operation in pyglet so the app stalls until all images are loaded.  
I am already caching images that have already been loaded previously but would probably need to offload the image loading to seperate thread but since it's only a few seconds I didn't bother.. :)