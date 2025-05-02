 This is a Model Context Protocol(MCP) Client that is able to use open-source LLM models to communicate with MCP Servers give access to tools that can be used by the LLM.

 Here are the instructions to run the MCP-Client:
 
 1. Clone the Repo

2. Install dependencies using UV pip install -r requirements.txt

3. Install Ollama Client from ollama.com and pull the LLM models you want to use with the client using the alternative chat client implementations.

4. If you have a Google Studio API Key, which you can obtain for free, update that to the '.env' file.

5. Run 'main'py' with UV run src/main.py

NOTE: I was unable to do a comprehensive test for a congenial LLM for tool calls. It is something that is quite hevily dependent upon the correct prompting. The present implementation of 'main.py' using Google Gen AI provided the best results so far as it has a very big context window. If you do want to try models other that google-gemini-2.5-pro you can use the other client implementations after changing their respective models file.
