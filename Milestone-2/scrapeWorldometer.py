#!/usr/bin/python3
import requests
from bs4 import BeautifulSoup
import csv

r = requests.get('https://www.worldometers.info/world-population/population-by-country/')
if r.status_code == 200:
    print("successfully retrieved HTML from worldometers")
    data = BeautifulSoup(r.text, 'html.parser')
    #print(data.table.thead.contents)
    body = data.table.tbody
    header = data.table.thead.tr
    colTitlesHTML = header.find_all('th')
    titles_string = list()

    for title in colTitlesHTML:
        titles_string.append(title.getText())

    bodyRows = body.find_all('tr')
    rowStrings = list()
    for row in bodyRows:
        contentCells = row.find_all("td")

        row_contents = list()  # The contents of each row.
        for cell in contentCells:
            cellContents = cell.getText()
            cellContents = cellContents.replace('%', '')
            cellContents = cellContents.replace(',', '')
            row_contents.append(cellContents)

        rowStrings.append(row_contents)

    with open("worldometerData.csv", 'w') as csvfile:
        csvWriter = csv.writer(csvfile, delimiter=',', dialect='excel', lineterminator="\n")
        csvWriter.writerow(titles_string)
        for row in rowStrings:
            csvWriter.writerow(row)


