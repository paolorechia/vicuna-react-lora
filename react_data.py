data = [
  {
    "instruction": "Find out how much 2 plus 2 is.",
    "output": """

Thought: I must use the Python shell to calculate 2 + 2
Action: Python REPL
Action Input: 
2 + 2  # line 1
Observation: 4

Thought: I now know the answer
Final Answer: 4
"""
  },
  {
    "instruction": "Write and execute a script that sleeps for 2 seconds and prints 'Hello, World'",
    "output": """

Thought: I should call the sleep function passing 2 as parameter
Action: Python REPL
Action Input: 
sleep(2)   # line 1
Observation: 

Thought: I should use the 'print' function to print 'Hello, World'
Action: Python REPL
Action Input: 
print('Hello, World')  # line 1
Observation: 

Thought: I now finished the script
Final Answer: I executed the following script successfully:

from time import sleep  # line 1
sleep(2)  # line 2
print('Hello, World')  # line 3
"""
  },
    {
    "instruction": "You have a variable age in your scope. If it's greater or equal than 21, say OK. Else, say Nay.",
"output": """

Thought: I should write an if/else block in the Python shell.
Action: Python REPL
Action Input:
if age >= 21:  # line 1
    # line 2 - this line has four spaces at the beginning, it's indented because of the if
    print("OK")  # line 3 - this line has four spaces at the beginning, it's indented because of the if
else:  # line 4
    # line 5 - this line has four spaces at the beginning, it's indented because of the else
    print("Nay") # line 6  - this line has four spaces at the beginning, it's indented because of the else


Observation: OK

Thought: I now know the answer
Final Answer: I have executed the task successfully.
"""},
    {
    "instruction": "Open a file and write 'Foo bar' to it.",
"output": """

Thought: I should create a Python code that opens a file and write 'Foo bar' it using the Python REPL
Action: Python REPL
Action Input:
with open("some_example_file.txt", "w") as fp: # line 1
    fp.write("Foo bar") # line 2 this has four spaces at the beginning, it's indented because of the with

Observation:

Thought: No errors were thrown by the code, and a file has been created with "Foor bar" inside.
Final Answer: I have executed the task successfully.
"""},
    {
    "instruction": "Open a HTTP connection using the requests.Session to the website https://somesite.com, and print the response status code.",
"output": """
Thought: I should use Python REPL to create the HTTP connection and print the response status code  
Action: Python REPL
Action Input:
import requests # line 1
with open(requests.Session() as session: # line 2
resp = session.get("https://sometime.com") # line 3
print(resp.status_code) # line 4

Observation: expected an indented block after 'with' statement on line 1 (<string>, line 3)

Thought: The lines after the 'with' statement have missing leading spaces. I should fix it.

Action: Python REPL
Action Input:
import requests # line 1
with open(requests.Session() as session: # line 2
    resp = session.get("https://sometime.com") # line 3 - this has four spaces at the beginning, it's indented because of the with
    print(resp.status_code) # line 4 - this has four spaces at the beginning, it's indented because of the with


Observation: 200

Thought: I now know the answer.
Final Answer: I have executed the task successfully.

"""},
    {
    "instruction": """
The website https://api.chucknorris.io/ describes in human readable HTML how to use a JSON API that generates jokes about Chuck Norris.
Write a script to:
1. fetch this website's HTML
2. save the results in a variable called 'response'
3. parse the HTML and inspect the contents, finding all links that could be used in a JSON API.

Hint: to parse the HTML use this snippet - there's no need to print it
import re
html = response.text
links = re.findall("https://api.chucknorris.io/[a-z]+/[a-z]+", html)

4. Find which one of the found links can be used to fetch a random joke about Chuck Norris. Look for one that contains the word 'random'
6. Use this link, e.g., fetch it
7. extract the joke from the response. Here's an example response to help you extract the joke:
    {
        "icon_url" : "https://assets.chucknorris.host/img/avatar/chuck-norris.png",
        "id" : "nvdxPp5BQK2iQWRbPxrHVg",
        "url" : "",
        "value" : "Chuck Norris died once. when greated by the grim reaper chuck proceded 2 round house kick the reaper.long story short death will never make that mistake again"
    } 
8. print the joke
9. inspect the joke result, and make a comment
10. finish
""",
"output": """
I should use the requests library to fetch the website's HTML
Action: Python REPL
Action Input:
response = requests.get('https://api.chucknorris.io/')

Observation: name 'requests' is not defined
Thought:I should import the requests library
Action: Python REPL
Action Input:
import requests

Observation: 
Thought:I should use the requests library to fetch the website's HTML
Action: Python REPL
Action Input:
response = requests.get('https://api.chucknorris.io/')

Observation: 
Thought:I should use the 'response.text' attribute to get the HTML content
Action: Python REPL
Action Input:
html = response.text

Observation: 
Thought:I should use the re library to find all links in the HTML
Action: Python REPL
Action Input:
import re
links = re.findall("https://api.chucknorris.io/[a-z]+/[a-z]+", html)

Observation: 
Thought:I should use the 'links' list to find the link that contains the word 'random'
Action: Python REPL
Action Input:
link = links[0]

Observation: 
Thought:I should use the 'requests' library to fetch the link
Action: Python REPL
Action Input:
response = requests.get(link)

Observation: 
Thought:I should use the 'json' library to parse the response
Action: Python REPL
Action Input:
response = response.json()

Observation: 
Thought:I should use the 'find' method to find the joke
Action: Python REPL
Action Input:
joke = response.find('value')

Observation: 'dict' object has no attribute 'find'
Thought:I should use the 'get' method to get the value
Action: Python REPL
Action Input:
joke = response.get('value')

Observation: 
Thought:I should use the 'json' library to print the joke
Action: Python REPL
Action Input:
print(json.dumps(joke))

Observation: name 'json' is not defined
Thought:I should import the json library
Action: Python REPL
Action Input:
import json

Observation: 
Thought:I should use the 'json' library to print the joke
Action: Python REPL
Action Input:
print(json.dumps(joke))

Observation: "Chuck Norris can hammer nails with a screwdriver."

Thought:I now know the final answer
Final Answer: "Chuck Norris can hammer nails with a screwdriver."
"""},
]