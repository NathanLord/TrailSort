## Create a virtual environment
```python -m venv venv```

## Activate the virtual environment
```.\venv\Scripts\activate```

### May need to temporarily bypass execution policy of power shell 

Do this before trying to activate

```Set-ExecutionPolicy -ExecutionPolicy Bypass -Scope Process```

<br><br> <br>

# Install Dependencies
```pip install -r requirements.txt```

## Install Flask
```pip install flask```

## Install TensorFlow
```pip install tensorflow```

# Install Cors
```pip install flask-cors```

# Install dotenv so flask can see the .env file
```pip install python-dotenv```

# Install tools for PostgreSQL connection
```pip install psycopg2-binary Flask-SQLAlchemy```


## Create Requirements
```pip freeze > requirements.txt```



# Create Folders to handle uploads

Create two folders called "uploads" and "processed"

# Make a '.env' file in your root of the backend 

This will hold where the location where you put your zip file that gets sent to the front end is. 
This will aslo hodl info about your database
Example:
```
PROCESSED_FOLDER=C:/Users/bob/Documents/thisCoolProject/BackEnd/processed

DATABASE_URL=postgresql://username:password@localhost:5432/your_database_name

```

# Run app

```python app.py```
