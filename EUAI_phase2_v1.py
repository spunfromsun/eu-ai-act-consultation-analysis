

def main() -> None:
    # %% Cell 1
    # NOTE: Jupyter magic: %pip install requests

    # %% Cell 2
    import requests
    import json

    # The URL you found
    api_url = "https://ec.europa.eu/info/law/better-regulation/api/allFeedback?publicationId=14488&keywords=&language=EN&page=0&size=10&sort=dateFeedback,DESC"

    try:
        # 1. Send a GET request to the API
        response = requests.get(api_url)
        
        # 2. Check if the request was successful (Status Code 200)
        if response.status_code == 200:
            # 3. Convert the response text directly into a Python Dictionary
            data = response.json()
            
            # Print the data to see what we got
            print("Success! Data retrieved.")
            print(json.dumps(data, indent=2)) # Pretty print the JSON
            
        else:
            print(f"Failed to retrieve data. Status Code: {response.status_code}")

    except Exception as e:
        print(f"An error occurred: {e}")

    # %% Cell 3
    https://ec.europa.eu/info/law/better-regulation/brpapi/feedback?publicationId=14488&page=0&size=50



if __name__ == "__main__":
    main()

