# Parachute Opening Shock
This repository houses python scripts and example codes to calculate parachute opening shock based on Pflanz and Moment-Impulse-Theorem Methods. The Opening Shock of a parachute is one of the most critical aspects of aerospace systems recovery design. If not dimensioned properly, it may lead to catastrophic failure of the end phase of the mission. 

Pflanz method has been extensively used in parachute design. Moment Impulse theorem for the OSCalc software is a more recent approach to estimating parachute opening loads. Both are semiempirical approaches. The OSCalc software executable is not easily found. Furtheremore, the Pflanz method has not been publically adapted into software. Therefore, when doing iterative design, specially in the initial phases of a mission, having to calculate SHock Loading bu hand is a tedius process. 

This code aims to alliviate that load, by allowing users to calculate shock forces from these two methods and streamlining the initial design stage. 

**In the future** I aim to upload tutorials on how to use this code, its limitations and more example calculations. 
This is an OpenSource project and you can use and modify the code however you please. I only ask for citation or acknowledgment.
My name is Carlos Albiñana Burdiel. You can text me on my [LinkedIn](www.linkedin.com/in/carlos-albibur) and email me at carlos.albibur@gmail.com
I will be happpy to help in any way.

### Background ###
I have worked on Sounding Rocket Recovery Systems for 4 years. I was the first Chief Recovery Engineer of the [Faraday Rocketry UPV Team](https://www.faradayupv.com/) and I later helped [Skyward Experimental Rocketry](https://skywarder.eu/) during my Masters. My bachelor thesis was precisely on Parachute Recovery Systems and their application on Rocket Recovery, which you can find in this [link](https://riunet.upv.es/entities/publication/81823cbf-da7a-43a7-80a3-b7a2dff80528)

## References
### Pflanz
The Pflanz method can be found in Knacke's Parachute System Design Manual. A free PDF version can be found in this [webpage](https://www.google.com/url?sa=t&source=web&rct=j&opi=89978449&url=https://apps.dtic.mil/sti/tr/pdf/ADA247666.pdf&ved=2ahUKEwi5_9uNx9qPAxVdRKQEHVamCfoQFnoECBMQAQ&usg=AOvVaw03cPmhqTwLBKAHpBRh4WUC)

### Moment-Impulse-Theorem (OSCalc)
The Moment-Impulse-Theorem is based on the OSCalc calculator. Reference material can be found in this [webpage](https://www.google.com/url?sa=t&source=web&rct=j&opi=89978449&url=https://cdn.imagearchive.com/rocketryforum/data/attach/414/414538-7152718.pdf&ved=2ahUKEwis9cSqx9qPAxVOUaQEHSawO4EQFnoECAkQAQ&usg=AOvVaw2BeOZhG88oq20Ru8_ZiUpH)

## Script description and use
The scripts you will find are:
- **Data_Extraction.py** Reads the txt file containing the values for Pflanz and OSCalc/MIT method. Outputs the coefficients of the interpolating functions and stores them in the pickle  file called _combined_data.pkl_. For **debug purposes**, ***there should be no need for you to run this script***
- **interpolating_functions.py** Evaluates the interpolating polynomials with the data from the pickle file. You can select to graph your results or just output the numerical result.
- **SingleValue_Example.py** Is just a code with an example calculation for a single output of force, for both Pflanz and MIT/OSCalc
- **RangeValues_Example.py** The same but you can input a probability density function for several inputs and obtain a range of values
- **combined_data.pkl** Python pickle file containing the coefficients for the interpolating functions 

