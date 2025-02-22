1. install abby finereader + license
2. install postgresql https://www.enterprisedb.com/downloads/postgres-postgresql-downloads
2.1 install pgvector https://github.com/pgvector/pgvector#windows
3. install python https://www.python.org/downloads/windows/
   check install for all
   select disk c root for directory
4. copy app files and scripts
   restore requirements.txt

5. process pdf with abbyy and place resulting files in respective dirs
5.1 Process files: 
   *a process files and add processing results to db
   *b add vectors to the relevant vector columns

6. install nssm https://nssm.cc/release/nssm-2.24.zip
   unpack and run win64 version
   - .\nssm.exe install IkonFlaskApp
   - In the Path field, ensure the path to python.exe is correct:
     Example: C:\Python3_13_2\python.exe
   - In the Startup directory field, set the directory containing script:
     Example: C:\project\scripts
   - In the Arguments field, provide the script name:
     Example: app_flask_v18.py
   - in Environment add HTTP_PLATFORM_PORT and OPENAI_API_KEY with values
  
7. Install Waitress
   
