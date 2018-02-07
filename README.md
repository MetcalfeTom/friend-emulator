# Friend-Emulator
Generating a NLP Model to replicate and replace my friends on WhatsApp

I wanted to see whether I could generate a neural network model to simulate the diction of my friends, by using the data of our group chat on WhatsApp.

WhatsApp allows you to export the 40000 most recent messages to a txt file.  Here's an excerpt from the raw file:
```
17/04/2015, 9:46 pm - James Hyde: <Media omitted>
17/04/2015, 11:32 pm - Alex Price: <Media omitted>
18/04/2015, 2:16 pm - Tom Metcalfe: <Media omitted>
18/04/2015, 2:50 pm - Joel Wilson: wat dat tommy
18/04/2015, 2:53 pm - Tom Metcalfe: 📱
18/04/2015, 2:56 pm - Joel Wilson: wat phone
18/04/2015, 3:00 pm - Tom Metcalfe: The OnePlus One, you can only get them on Tuesdays
18/04/2015, 3:02 pm - Joel Wilson: seems confusing
18/04/2015, 3:11 pm - Tom Metcalfe: ❌clusive
18/04/2015, 3:11 pm - Alex Price: Oh my god
18/04/2015, 3:11 pm - Alex Price: You are a legend. Does it feel exclusive?
18/04/2015, 3:12 pm - Alex Price: I want one!!!!
18/04/2015, 3:19 pm - Tom Metcalfe: <Media omitted>
18/04/2015, 3:20 pm - Tom Metcalfe: Also it's huge...almost too difficult to type with one hand.
18/04/2015, 3:43 pm - Joel Wilson: yh I got i6 and its muad
18/04/2015, 4:40 pm - Tom Metcalfe: Since when?
18/04/2015, 4:57 pm - Joel Wilson: like this week
18/04/2015, 10:18 pm - Liam Denison: What happened to your nexus?
18/04/2015, 10:28 pm - Tom Metcalfe: Charging connector died R.I.P.
18/04/2015, 10:28 pm - Tom Metcalfe: Ordered a new battery and charging port from eBay to fix it but nothing worked
18/04/2015, 10:29 pm - Tom Metcalfe: Funeral was on Monday
```

# Pre-processing
Firstly preprocessing to remove the "<Media omitted>" messages, timestamps, categorize by user
