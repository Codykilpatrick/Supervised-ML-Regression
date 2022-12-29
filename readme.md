## Predicting market prices in the MMORPG Eve online with supervised machine learning

![Scatterplot for results of Ridge Regression](https://i.imgur.com/rQTj8sy.png)

## Background information 🚀
Eve Online is a Massive Multiplayer Online Role Playing Game. Players are set in the far future and tasked with piloting spaceships to make money (ISK) and do whatever they desire. Players are able to come together to form corporations, then corporations of like minded players form alliances. Eve online has a completely player driven market, that means everything is created and destroyed within the game with very few external injections.

## Problem description 👨‍💻
The main objective of this project to see if I could predict the average price of an item and use that information for buying or selling at certain times to make a profit.

## Features of the data 📀
The dataset has the following features:
- volume: The ammount that specific item was traded in one region
- order_count: How many individual buy and sell requests were placed on the market
- lowest: The lowest price an item sold for
- average: The average price an item sold for
- highest: The highest price an item sold for
- highlow_difference: The difference between the highest and item sold for and the lowest.
- type_id: The specific ID associated with each unique item.
- typename: The ingame name of each item correlating to the type_id
- category_id: The specific code associated with each unqiue category of item
- categoryname: The classification name that encompasses all group names.
- group_id: The specific code associated with each unique group.
- groupname: The classification name that encompasses all group names.

## Models used 🦾
- Linear Regression
- Lasso Regression
- Ridge Regression
- ElasticNet CV

## Technologies used 💾
- Jupyter Notebook
- Python
- Git

## Attributions 👥
- CCP games
- everef.net

## [Link to final report](https://docs.google.com/document/d/1RkVIY1f_A61r1zoL2eDe_r9xzNFJhfKKFKhm9LO14bk/edit)