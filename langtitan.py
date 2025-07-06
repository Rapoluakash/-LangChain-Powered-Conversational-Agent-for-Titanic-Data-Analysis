import os
import warnings
import pandas as pd
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain.llms import OpenAI

# Suppress warnings
warnings.filterwarnings("ignore")

# Set your OpenAI API Key
os.environ["OPENAI_API_KEY"] = "Enter_your_api_key_here"

# Load the Titanic dataset
df = pd.read_csv(r"C:\\Users\\rapol\\Downloads\\DATA SCIENCE\\3. Nov\\11th - ML\\TITANIC")
print(df.shape)
print(df.head())

# Initialize the OpenAI LLM
llm = OpenAI()

# Create the agent with the first DataFrame
agent = create_pandas_dataframe_agent(
    llm=llm,
    df=df,
    verbose=True,
    allow_dangerous_code=True  # ✅ REQUIRED for full execution
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
