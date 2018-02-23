# Friend-Emulator

More & more of my friends are moving away from where I live, some across different time zones.  To keep myself from becoming lonely, I figured the next best thing to physical human contact is a natural language processing bot that can emulate the style and semantics of my friends...

## Getting & processing the data

Thankfully WhatsApp allows you to export group chats as an e-mail, which provided me with 20024 unique messages across 13 of my friends as a text file.  Unfortunately, it needed a little preprocessing to get into a suitable format for analysis

As I'm not one to re-invent the wheel here, journocode has published an [amazing script for parsing the raw text data into a CSV file,](https://github.com/journocode/datavizwhatsapp) and with a bit of tinkering I was able to get the messages in a nice format for analysis with pandas and seaborn

## Exploratory Data Analysis

First off, a click visualisation of user activity

![alt text](https://github.com/MetcalfeTom/Friend-Emulator/blob/master/UserPieChart.png?raw=true "User activity" =300x300)
