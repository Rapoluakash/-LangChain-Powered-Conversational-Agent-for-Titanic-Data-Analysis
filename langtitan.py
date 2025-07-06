
import os

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("❌ OPENAI_API_KEY is missing. Set it in .env or GitHub Secrets.")
print("✅ API key found.")
os.environ["OPENAI_API_KEY"] = api_key

import warnings
import pandas as pd

# ✅ Try importing latest supported OpenAI LLM wrapper
try:
    from langchain_community.llms import OpenAI
except ImportError:
    from langchain.llms import OpenAI  # fallback for older versions

from langchain_experimental.agents import create_pandas_dataframe_agent

# ✅ Get OpenAI API key from environment (GitHub Secret or .env)
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("❌ OPENAI_API_KEY is missing. Set it in .env or GitHub Secrets.")
print("✅ API key found and loaded from environment.")
os.environ["OPENAI_API_KEY"] = api_key

# Suppress warnings
warnings.filterwarnings("ignore")

# Load the Titanic dataset
df = pd.read_csv("https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv")
print(df.shape)
print(df.head())

# Initialize the OpenAI LLM
llm = OpenAI()

# Agent with first DataFrame
agent = create_pandas_dataframe_agent(
    llm=llm,
    df=df,
    verbose=True,
    allow_dangerous_code=True
)

print(agent.run("How many rows are there"))
print(agent.run("How many people have more than 3 siblings"))

# Create second copy and fill missing Age values
df1 = df.copy()
df["Age"] = df1["Age"].fillna(df1["Age"].mean())

agent = create_pandas_dataframe_agent(
    llm=llm,
    df=df,
    df1=df1,
    verbose=True,
    allow_dangerous_code=True
)

print(agent.run("How many rows in the age column are different"))

# Create third copy and add derived column
df2 = df1.copy()
df2["Age_Multiplied"] = df1["Age"] * 2

agent = create_pandas_dataframe_agent(
    llm=llm,
    df=df,
    df1=df1,
    df2=df2,
    verbose=True,
    allow_dangerous_code=True
)

print(agent.run("Are the number of columns same in all the dataframes"))
