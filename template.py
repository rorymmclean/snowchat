from langchain.prompts.prompt import PromptTemplate
from langchain_core.prompts import ChatPromptTemplate

table_info = """
Context - (Schema Details):

** Table 1: medals_data ** (This table stores blueprint information)

This table contains the data for all blueprints stored in the system. The table includes the following fields:

-  id (integer value that represents the primary key for the record)
-  doc_nbr (varchar(30) value that represents a document number for the record)
-  part_nbr (varchar(30) value that represents a part number for the record)
-  national_stock_number (varchar(30) value that represents the assigned national stock number for this record. May also be referred to as "NSN".)
-  niin (varchar(30) value that represents the assigned "National Item Identification Number" for this record.)
-  cage (varchar(30) value represents the "Commercial and Government Entity Code" assigned to the vendor providing this part.)
-  title (varchar(1024) value that describes the part provided by this record)
-  photo_data_1 (bytea value contains an image related to the blueprint)


** Table 2: niin_catalog ** (Table stores niin metadata)

This table contains the metadata details for niin codes. The table includes the following fields:

-  niin (varchar(30) value that represents the "National Item Identification Number" code for this record. It is joined to the niin column in the medals_data table.)
-  product_name (varchar(255) value is the name of the niin record.)
-  width (numeric value is the width of the product in inches.)
-  height (numeric value is the height of the product in inches.)
-  depth (numeric value is the depth of the product in inches.)
-  weight (numeric value is the weight of the product in pounds.)


** Table 3: vendors ** (Table stores vendor metadata)

This table contains the vendor metadata details for cage controls. The table includes the following fields:

-  cage (varchar(30) value represents the "Commercial and Government Entity Code" assigned to the vendor providing this part. Column used for joining to the cage column in the medals_data table.)
-  org_name (varchar(255) value is the name of the vendor)
-  street (varchar(1024) value is the street address of the vendor)
-  city (varchar(60) value is the city of the vendor)
-  state (char(2) value is the state of the vendor. The state is in the 2 character postal standard.)
-  zip (char(5) value is the zipcode of the vendor)
-  phone (varchar(15) value is the phone of the vendor)

"""

clarifier_template = """Carefully consider the user's question and determine which case is true.

1) IF the user's question is unrelated to an existing conversation, simply respond with the 
original question without any modifications or additional text.

2) IF the user's question is a continuation of an existing conversation, use previous question in the 
Chat History section to rewrite the user's current question into a new, single question that 
incorporates all the information the user wants in their output.
** You are not asking the user a question. You are using history to clear up any ambiquity. **
Example #1:
   - Previous User Question: "What is the total weight for blueprints from virginia companies?"
   - Current User Question: "Break this down further by vendor name"
   - AI Rewrite of Question: "What is the total weight for blueprints from virginia companies broken down by vendor name?"
Example #2:
   - Previous User Question: "How many blueprints result in products over 20 pounds?"
   - Current User Question: "Are any of these from virginia companies?"
   - AI Rewrite of Question: "How many blueprints from Virginia companies result in products over 20 pounds"


Only return either the original or the rewritten question. Do not include any additional text. 

<Chat History>
{chat_history}
</Chat History>

<QUESTION>
{question}
</QUESTION>

Assistant:"""

clarifier_prompt = PromptTemplate.from_template(clarifier_template)

context_template = """ 
You're an AI assistant specializing in data analysis with Postgres SQL. 
When providing responses, write in a friendly and a conversational tone.

If the user only asks about your capabilities, provide a general overview of your ability to assist with 
data analysis tasks using SQL, instead of writing a specific SQL queries. 

The CONTEXT is provided to you as a reference to generate SQL code.

Expected Scenario:
If the USER_QUESTION provided pertains to data analysis or SQL tasks, generate SQL code 
based on the CONTEXT provided. Make sure that your SQL is compatible with a Postgres database and 
is deliniated the code using "```sql\n" at the start of your code and "\n```" after your code. 
When you perform a join, be sure to use a table alias on all columns in your SQL code.
All columns in your SELECT statement must be uniquely named to avoid duplications.
In addition to your code, write a one paragraph explaination of how you arrived at the SQL code.
** Your response should only include the code and the one paraphraph explanation. **
** Do not include a "<USER_QUESTION>" nor a "<CONTEXT>" section in your response. **
Do not use a "*" in your select statement. Always list the columns specifically.
If the required column isn't explicitly stated in the CONTEXT, suggest an alternative using 
available columns, but do not assume the existence of any columns that are not mentioned. 
Also, do not attempt to modify the database in any way (no insert, update, or delete operations). 
You are only allowed to query the database. 
** Do not include a "<USER_QUESTION>" nor a "<CONTEXT>" section in your response.**

Other Possible Scenarios:
- If the USER_QUESTION does not require a SQL statement, respond appropriately without generating SQL queries. 
- When the USER_QUESTION expresses gratitude such as saying "Thanks", interpret it as a signal to conclude the conversation and 
respond with an appropriate response without generating another SQL query.  
- If you don't know the answer, simply state, "I'm sorry, I don't know the answer to your question."

Write your response in markdown format.
Do not worry about access to the database.

<USER_QUESTION>
{standalone_question}
</USER_QUESTION>

<CONTEXT>
"""+table_info+"""
</CONTEXT>

Assistant:
"""

SQL_prompt = PromptTemplate.from_template(context_template)

eval_template = """ 
Consider how to answer the "User Question". You have access to a database described under the Context section. 
You can take three approaches to answering the user's question.

1) You can use a semantical similarity search against the database when the USER QUESTION asks for
records or blueprints "similar to", "dealing with", or "about" a topic. Respond with "KNN". Use this method 
rather than performing "LIKE" or wildcard SQL searches.

2) If the question can be answered using a SQL query, respond with "SQL". 
Avoid using "LIKE" and wildcard searches. 

3) If the question is unrelated to the database, respond with "NA".

Do not respond with any other text except one of these three answers.

<USER_QUESTION>
{standalone_question}
</USER_QUESTION>

"""+table_info+"""


Assistant:
"""

eval_prompt = PromptTemplate.from_template(eval_template)

analysis_template = """ 
Review the data in the <DATA> section and write a brief but concise analysis of the data.
Call out any trends, correlations, anamolies, or any other interesting patterns in the data.
Write up your analysis in a friendly but business-like manner. 

<DATA>
{mydata}
</DATA>

Assistant:
"""

analysis_prompt = PromptTemplate.from_template(analysis_template)
