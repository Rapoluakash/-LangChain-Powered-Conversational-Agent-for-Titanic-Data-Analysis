import os
import warnings
import pandas as pd
from langchain_experimental.agents import create_pandas_dataframe_agent

# ✅ Try importing from langchain-community (to fix future deprecation)
try:
    from langchain_community.llms import OpenAI
except ImportError:
    from langchain.llms import OpenAI  # fallback if not installed

# ✅ Load from .env if it exists (for local dev)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # ignore if not using dotenv

# ✅ Set OpenAI API key securely
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("❌ OPENAI_API_KEY is missing. Set it in .env or GitHub Secrets.")
os.environ["OPENAI_API_KEY"] = api_key

# Suppress warnings
warnings.filterwarnings("ignore")

# Load the Titanic dataset
df = pd.read_csv("https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv")
print(df.shape)
print(df.head())

# Initialize the OpenAI LLM
llm = OpenAI()

# Create the agent with the first DataFrame
agent = create_pandas_dataframe_agent(
    llm=llm,
    df=df,
    verbose=True,
    allow_dangerous_code=True
)

# Run some basic queries
print(agent.run("How many rows are there"))
print(agent.run("How many people have more than 3 siblings"))

# Create a second copy of the DataFrame and fill missing age values
df1 = df.copy()
df["Age"] = df1["Age"].fillna(df1["Age"].mean())

# Create agent with two DataFrames
agent = create_pandas_dataframe_agent(
    llm=llm,
    df=df,
    df1=df1,
    verbose=True,
    allow_dangerous_code=True
)

print(agent.run("How many rows in the age column are different"))

# Create third copy and add new column
df2 = df1.copy()
df2["Age_Multiplied"] = df1["Age"] * 2

# Final agent with three DataFrames
agent = create_pandas_dataframe_agent(
    llm=llm,
    df=df,
    df1=df1,
    df2=df2,
    verbose=True,
    allow_dangerous_code=True
)

print(agent.run("Are the number of columns same in all the dataframes"))
