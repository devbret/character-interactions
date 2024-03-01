# Character Interactions

As a brief summary, with this program, users rely on Python and D3.js to map direct conversations between different characters in a body of text.

## Installation Instructions

**Please Note:** *The following instructions assume that you are running Virtual Studio Code on a Linux Operating System. But may still be generally, otherwise applicable to Windows Operating Systems, as well as other IDEs on Linux.*

**Please Also Note:** *There are no guarantees with this program. And it will likely, at this stage of development, produce inaccuracies.* 

First, create a directory on your local machine, which will be home to all of the related files. You can name this folder whatever you would like to. Then download and save the Python and HTML files found within this GitHub repo, to your new directory.

Open a terminal and make sure that you have the spaCy library installed.

Your source .txt file, can contain any text. and it is best if that file is also located in your newly created directory. 

Once you have all of files organized in the same directory, open the app.py file, and edit line 46 to reflect the name of your .txt file. Then save that change.

Open a terminal from your new directory, assuming your are running Python 3+, and enter **python3 app.py**, and press enter. That will eventually produce a JSON file that will be used by the HTML file to visualize the information.

Open VS Code and start a local web development server. This should launch the index.html file, which will then read the local JSON file, which was just previous created. And if everything is working correctly, should be able to find a visualization on your screen. Although, the visualization might be off to the right of your screen, which can be found by scrolling horizontally.

## Use Cases

The biggest and most significant use case for this program is understanding if two named parties can directly communicated within a body of text. You can also observe degrees of separation with the associated visualization.

## Other Notes

This program is only designed to work with .txt text-based files. But can conceivably work with any amount of data. And was built for digesting entire books at once.
