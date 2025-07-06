
#🧠 LangChain-Powered Conversational Agent for Titanic Data Analysis

This project demonstrates how to build a conversational AI agent using **LangChain**, **OpenAI**, and **Pandas** to interact with the famous **Titanic dataset** using natural language queries.

Instead of writing SQL or DataFrame code, just ask your questions like:
> "How many passengers had more than 3 siblings?"  
> "What is the average age after filling missing values?"  
> "Are there any negative values in the Age_Multiplied column?"

---

#🚀 Features

- Load Titanic dataset from GitHub
- Fill missing values in the `Age` column
- Generate a new column `Age_Multiplied` for analysis
- Use `langchain_experimental`'s `pandas_dataframe_agent` to run natural language queries
- Integrate OpenAI LLM via `langchain.llms.OpenAI`

---

#📁 Project Structure

```

├── langtitan.py            # Main Python script
├── requirements.txt        # pip dependencies
├── environment.yml         # Optional: conda environment
├── .env                    # OpenAI API Key (not uploaded to GitHub)
├── .gitignore              # Git ignored files
├── README.md               # You're reading this
└── .github/workflows/      # GitHub Actions CI/CD

```

---

#⚙️ Setup Instructions

#🔐 1. Set your OpenAI API key

Create a `.env` file in the root of the project:
```

OPENAI\_API\_KEY=your\_openai\_api\_key\_here

````

> ✅ Use [python-dotenv](https://pypi.org/project/python-dotenv/) to load this key automatically.

---

### 📦 2. Install dependencies

#### Option A: Using pip
```bash
pip install -r requirements.txt
````

#Option B: Using conda

```bash
conda env create -f environment.yml
conda activate langchain-titanic
```

---

#▶️ 3. Run the script

```bash
python langtitan.py
```

---

#🤖 Example Queries

Here are some natural language queries you can run:

* "How many rows are there?"
* "How many people had more than 3 siblings?"
* "How many rows have different original and filled ages?"
* "What is the average of the Age\_Multiplied column?"

---

#📊 Dataset

This project uses the **Titanic dataset** from [Data Science Dojo](https://github.com/datasciencedojo/datasets):

```
https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv
```

---

#📄 License

This project is licensed under the [MIT License](LICENSE).

---

#🙋‍♂️ Author

**Rapolu Akash**
📧 [rapoluakash3@gmail.com](mailto:rapoluakash3@gmail.com)
🔗 [LinkedIn](https://linkedin.com/in/akash-rapolu-67043a344)

---

#🌟 Star this repo if you like it!

```
git clone https://github.com/your-username/langchain-titanic-agent.git
cd langchain-titanic-agent
```

Happy querying! 🧠📊

```

---

Let me know if you want me to:
- Generate a project thumbnail image for GitHub
- Write the `.gitignore` for this project
- Help you publish this on Hugging Face or Streamlit

Would you like the README in Markdown preview format (e.g., for testing locally)?
```
