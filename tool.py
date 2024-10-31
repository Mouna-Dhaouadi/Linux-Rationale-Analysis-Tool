import streamlit as st
import pandas as pd
import asyncio
import numpy as np
import base64
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from helpers import *
import time
import requests
import json
import csv
from datetime import datetime
from urllib.parse import urlparse, parse_qs
import streamlit as st
import time



# import h5py
# f = h5py.File('models\\bi_lstm_model_rationale.h5', 'r') 
# print(f.attrs.get('keras_version')) # keras 2.15.0 
# current: Name: keras: 3.6.0 


# Function to generate the download link
def get_download_link(file_path, file_name, button_text):
    # Read the file content
    with open(file_path, "rb") as file:
        file_content = file.read()

    # Encode the file content in base64
    b64_file = base64.b64encode(file_content).decode()
    # Create the HTML download link
    download_link =  f"""
    <a href="data:application/octet-stream;base64,{b64_file}" download="{file_name}">
        <button style=\"
            border: solid 1px lightgrey;
            background-color: white;
            color: black;
            padding: 7px;
            text-align: center;
            display: inline-block;
            font-size: 16px;
            cursor: pointer;
            border-radius: 10px;
            margin-bottom:10px;
        \">{button_text}</button>
    </a>
    """
    return download_link


async def run_async_word_cloud(placeholder, text_list, df, option, L):
    # Generate the word cloud asynchronously
    img = await word_clouds(df, text_list, option, L)
    # Display the word cloud image in the placeholder
    placeholder.pyplot(img)  


async def run_async_factors_commit_size(placeholder, df, commits_length_list, commits_rationale_density_list, commits_IDs_list, option ):
    img = await factors_commit_size(df, commits_length_list, commits_rationale_density_list, commits_IDs_list, option )
    file_path = "figures/factors__commit_size__"+ option +'.pdf'
    file_name = "factors__commit_size__"+ option +'.pdf'
    download_link_html = get_download_link(file_path, file_name, "Download Figure")
    with placeholder.container():
      st.pyplot(img) 
      st.markdown(download_link_html, unsafe_allow_html=True)

        
async def run_async_factors_developers(placeholder, dff4, commits_length_list, commits_rationale_density_list, commits_IDs_list, x,y, option ):
    img = await factors_developers(dff4, commits_length_list, commits_rationale_density_list, commits_IDs_list, x, y , option)
    file_path = "figures/factors__developers__"+ option +'.pdf'
    file_name = "factors__developers__"+ option +'.pdf'
    download_link_html = get_download_link(file_path, file_name, "Download Figure")
    with placeholder.container():
      st.pyplot(img) 
      st.markdown(download_link_html, unsafe_allow_html=True)


async def run_async_commit_structure(placeholder, decision_positions, rationale_positions, option):
  img = await commit_structure(decision_positions, rationale_positions, option)
  file_path = 'figures\commit_strcuture_normalized_'+ option +'.pdf'
  file_name = 'commit_strcuture_normalized_'+ option +'.pdf'  
  download_link_html = get_download_link(file_path, file_name, "Download Figure")
  with placeholder.container():
    st.pyplot(img) 
    st.markdown(download_link_html, unsafe_allow_html=True)


async def run_async_rationale_evolution(placeholder, dff6_y, option, start):
  img = await rationale_evolution(dff6_y, option)
  file_path = 'figures\evolution_decision_rationale_'+ option +'.pdf'
  file_name = 'evolution_decision_rationale_'+ option +'.pdf'
  download_link_html = get_download_link(file_path, file_name, "Download Figure")
  with placeholder.container():
    st.pyplot(img) 
    st.markdown(download_link_html, unsafe_allow_html=True)
  
  end = time.time()
  print('Execution Time:', end - start)

################################### MAIN
async def main():
  st.header("Module Rationale Analyses")

  # SLOB_FILE = 'prediction_results_on_the_slob.csv'
  # BUTTON_FILE = 'prediction_results_on_the_button.csv'
  # FS_FILE = 'prediction_results_on_the_fs.csv'
  # MIGRATE_FILE = 'prediction_results_on_the_migrate.csv'
  # migrate api: https://api.github.com/repos/torvalds/linux/commits?path=mm/migrate.c \n  
####################################
  url = st.text_input("Enter the API URL of the module", "")

  st.markdown(""" API URL examples: \n 
    Linux Kernel \n 
    http://api.github.com/repos/torvalds/linux/commits?path=mm/oom_kill.c \n
    https://api.github.com/repos/torvalds/linux/commits?path=mm/slob.c \n 
    https://api.github.com/repos/torvalds/linux/commits?path=fs/fsopen.c \n 
    https://api.github.com/repos/torvalds/linux/commits?path=drivers/acpi/button.c \n 
    Django \n
    https://api.github.com/repos/django/django/commits?path=django/forms/forms.py \n 
    React \n
    https://api.github.com/repos/facebook/react/commits?path=packages/react-reconciler/src/ReactFiber.js \n
    Node.js \n
    https://api.github.com/repos/nodejs/node/commits?path=lib/stream.js \n
    """ ) 

  credentials = st.radio(
    "Github credentials",
    ["Use Mine", "Use Mouna's"],
  )
  if credentials == "Use Mouna's":
    username = st.secrets["github_username"]
    token = st.secrets["github_token"]
  else:
    username = st.text_input("Enter your Github Username", "")
    token = st.text_input("Enter your Github API Token", "")  
    url_doc= "https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/managing-your-personal-access-tokens#creating-a-personal-access-token-classic"
    st.markdown("See how to [create a Github token](%s)" % url_doc) 
  

  ##### Button Clicked
  if st.button("Start Module Analysis", disabled=not bool(url.strip()) ): 

    start = time.time()

    # Parse the URL and extract query parameters
    parsed_url = urlparse(url)
    query_params = parse_qs(parsed_url.query)
    # Get the 'path' parameter
    option_display = query_params.get('path', [None])[0]
    option=option_display.replace('/', '_').replace('.c', '')

    #TODO: if prediction file exsist, skip this

    ################################ PREPARE DATASET
    with st.spinner('Getting Commit Messages'):
      # Fetch all commits
      all_commits = fetch_commits(url, username, token)

      # Serializing json
      json_object = json.dumps(all_commits, indent=4)
    
      # Writing to sample.json
      with open("sample_"+option+".json", "w") as outfile:
            outfile.write(json_object)
      
      # create a csv sheet with commit ID/node_ID , commit author name/email, commit commiter name/email, commit message on each line 
      with open("sample_"+option+".json") as json_file:
          commits = json.load(json_file)

    with st.spinner('Preparing Commit Messages'):
      # now we will open a file for writing
      data_file = open('data\data_file_'+option+'.csv', 'w', encoding="utf-8", newline='')
    
      # create the csv writer object
      csv_writer = csv.writer(data_file)

      # Writing headers of CSV file
      header = ["commit ID", "author name", "committer name", "message", "URL", "commit date"] 
      csv_writer.writerow(header)
      
      non_merge_commits = 0 
      for comm in commits:
            # Writing data of CSV file + remove merge commit
            if not str(comm["commit"]["message"]).startswith("Merge branch") and not str(
                    comm["commit"]["message"]).startswith(
                    "Merge tag"):
                csv_writer.writerow([comm["node_id"], comm["commit"]["author"]["name"], comm["commit"]["committer"]["name"],
                                    comm["commit"]["message"], comm["commit"]["url"],  datetime.strptime(comm["commit"]['author']['date'][:10], "%Y-%m-%d").date() ])
                non_merge_commits = non_merge_commits + 1 
      
      data_file.close()

      print(non_merge_commits)

    with st.spinner('Preprocessing Commit Messages'):
      df = pd.read_csv('data\data_file_'+option+'.csv', encoding="utf-8")
      df['message_preprocessed'] = df.apply(preprocess, axis=1)
      df = df.explode('message_preprocessed')
      
      # convert DF to CSV
      #created_file = convert_df(df)
      df.to_csv('data_file_'+ option + '_preprocessed.csv')

    ######################### CLASSIFICATION

    ########  get the classifiers
    with st.spinner('Classification - Loading models'):
      model_rationale, model_decision = load_models()

    with st.spinner('Classification - Preparing Tokenizer'):
      ### prepare tokenizer: same preprocessing  as training
      df = pd.read_csv('data/dataset_3_labels_merged.csv', sep=';')
      texts = df['message_preprocessed']
      dataf = pd.DataFrame({'text': texts})
      max_words = 1000
      tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
      tokenizer.fit_on_texts(dataf['text'])
      vocab_size = len(tokenizer.word_index) + 1
      sequences = tokenizer.texts_to_sequences(dataf['text'])
      max_sequence_length = max(len(seq) for seq in sequences)

    with st.spinner('Classification - Getting preprocessed data'):  
      ###### get preprocessed data
      df = pd.read_csv('data_file_'+ option +'_preprocessed.csv' )
      X_test =  df['message_preprocessed']

    with st.spinner('Classification - Applying classifiers'):
      ###### apply the calssifiers
      # Convert text to sequences
      sequences = tokenizer.texts_to_sequences(X_test.astype(str))
      padded_sequences = pad_sequences(sequences, maxlen=max_sequence_length)
      # Making Decision predictions
      y_pred_prob = model_decision.predict(padded_sequences) 
      y_pred = (y_pred_prob > 0.5).astype(int)  # Threshold probabilities to get binary predictions
      y_pred = y_pred.flatten()
      
      df_predicted = pd.DataFrame({'commit ID': df['commit ID'],
                                  'text': X_test, 
                                  'author name':df['author name'], 
                                  'commit date': df['commit date'],
                                  'predicted_decision':y_pred
                                  })

      # Making Rationale predictions
      y_pred_prob = model_rationale.predict(padded_sequences) 
      y_pred = (y_pred_prob > 0.5).astype(int)  # Threshold probabilities to get binary predictions
      y_pred = y_pred.flatten()
      df_predicted['predicted_rationale'] = y_pred

    with st.spinner('Classification - Saving the results'):
      file_name = 'prediction_results_on_the_' + option +'.csv'  
      file_path = "data\\" + file_name 
      df_predicted.to_csv(file_path, sep=';', index=False)


    ################################ VISUALIZE DATASET
    if file_path is not None:

      st.header('The ' + option_display + ' module')


      df = pd.read_csv(file_path, sep=';', encoding ='latin-1')
      df = df.rename( columns = { 'text' :'message_preprocessed' , 
                                'predicted_rationale' : 'Rationale' , 
                                'predicted_decision' : 'Decision' } )
      
      st.markdown('Resulting dataset:')
      st.write(df)

      #TODO: make appear when file ready
      download_link_html = get_download_link(file_path, file_name, "Download CSV file")
      st.markdown(download_link_html, unsafe_allow_html=True)


      commits_IDs_list = list (  dict.fromkeys(  df['commit ID']) ) #remove duplicates
      st.write( "Number of commits: " + str(len(commits_IDs_list)))
      st.write( "Number of sentences: " + str(len(df)))


      ################################ DISTRIBUTION
      st.subheader("Distribution")
      decision_only_sentences, rationale_only_sentences, _, _ = distribution_categories(df)
      #TODO: Venn diagram

      ################################ WORD CLOUDS
      st.subheader("Word Clouds")

      col1, col2= st.columns(2)
      
      with col1:
        st.markdown("Decision only sentences")
        placeholder1 = st.empty() 
        with placeholder1 : st.markdown(
              """<div style='text-align:center; font-size:24px; color:gray;'>Loading... ⏳</div>""",
              unsafe_allow_html=True,
          )
        asyncio.create_task(run_async_word_cloud(placeholder1, decision_only_sentences, df, option, "D"))
        #word_clouds(df, decision_only_sentences)

      with col2:
        st.markdown("Rationale only sentences")
        placeholder2 = st.empty()   
        with placeholder2 :st.markdown(
              """<div style='text-align:center; font-size:24px; color:gray;'>Loading... ⏳</div>""",
              unsafe_allow_html=True,
          )     
        #word_clouds(df, rationale_only_sentences)
        asyncio.create_task(run_async_word_cloud(placeholder2, rationale_only_sentences, df, option, "R"))



      #TODO: add download image buttons 

      

      ############################ ANALYSES ; presence - abundance
      st.subheader("Rationale Presence")

      commits_that_contain_rationale_list=[]

      for comit_id in  commits_IDs_list :
        if commit_contains_rationale(comit_id, df):
        #  print(comit_id)
          commits_that_contain_rationale_list.append(comit_id)

      total_number_of_commits = df['commit ID'].nunique()
      number_of_commits_that_contain_rationale = len(commits_that_contain_rationale_list)
      rationale_percentage = (number_of_commits_that_contain_rationale / total_number_of_commits ) #* 100
      st.write( "Total Number of commits: " + str(total_number_of_commits))
      st.write( "Number of commits that contain rationale: " + str(number_of_commits_that_contain_rationale))
      st.write( f"Rationale Percentage: {rationale_percentage:.2%}")
      # st.write( "Rationale Percentage: " + str( rationale_percentage ))
      with st.expander("See explanation"):
        st.latex(r'''
          Rationale \ Percentage = \frac { 
              number \ of \ commits \ that \ contain \ rationale}
              {total \ number \ of \ commits}   
          ''')


      ############################ ANALYSES :   presence - amount

      commits_rationale_density_list=[]
      for comit_id in  commits_IDs_list :
        commits_rationale_density_list.append(rationale_density(comit_id, df))
      
      commits_length_list = []
      for comit_id in  commits_IDs_list :
        commits_length_list.append(commit_size(comit_id, df))
    
      average_rationale_density = sum (commits_rationale_density_list) / number_of_commits_that_contain_rationale
      st.write( f"Average Rationale Density: {average_rationale_density:.2f}"  )
      with st.expander("See explanation"):
        st.latex(r'''
         Commit \ Rationale \ Density = \frac {number \ of \ sentences \ labelled \ as \  Rationale }
         {total \ number \ of \ sentences \ in \ a \ commit}
            ''')
        st.latex(r'''
          Average \ Rationale \ Density = \frac { 
            \sum { commit \  rationale \ density  } 
            }
            {number \ of \ commits \ that \ contain \ rationale}   
          ''')


    ############################ ANALYSES :   factors - commit size
      st.subheader("Rationale Factors ")

      col1, col2= st.columns(2)
      
      with col1:
        st.markdown("Rationale density vs Commit size")
        placeholder3 = st.empty()        
        with placeholder3 : st.markdown(
              """<div style='text-align:center; font-size:24px; color:gray;'>Loading... ⏳</div>""",
              unsafe_allow_html=True,
        )
        asyncio.create_task(run_async_factors_commit_size(placeholder3, df, commits_length_list, commits_rationale_density_list, commits_IDs_list, option ))


    ############################ ANALYSES :   factors - Developers experience
      authors_list = []
      for comit_id in commits_IDs_list:
        rslt_df = df[df['commit ID'] == comit_id]
        author =   rslt_df.iloc[0]['author name']
        authors_list.append(author)

      df_rq4 = pd.DataFrame(
        {'comit_ID': commits_IDs_list,
        'author': authors_list,
        'comit_rationale_density': commits_rationale_density_list
        })
      
      df_rq4['Number of commits'] = df_rq4.groupby('author')['author'].transform('count')
      dff4 = df_rq4.groupby(['author'], as_index=False).mean(numeric_only=True)
      dff4['Number of commits'] = dff4['Number of commits'].apply(lambda x: int(x))

      x = dff4['Number of commits']
      y = dff4['comit_rationale_density']

      with col2:
        st.markdown("Rationale density vs Developers Experience")
        placeholder4 = st.empty()        
        with placeholder4 : st.markdown(
              """<div style='text-align:center; font-size:24px; color:gray;'>Loading... ⏳</div>""",
              unsafe_allow_html=True,
        )
        asyncio.create_task(run_async_factors_developers(placeholder4, dff4, commits_length_list, commits_rationale_density_list, commits_IDs_list, x,y, option ))
        #factors_developers(dff4, commits_length_list, commits_rationale_density_list, commits_IDs_list, x,y )

    
    ############################ ANALYSES :   Message Structure
      st.subheader("Commit Message Structure ")
      decision_positions = []
      rationale_positions = []

      for commitID in commits_IDs_list:

        c_df = df[df['commit ID'] == commitID]
        c_df = c_df.reset_index()

        size = len(c_df)
        for i in range(size):

          if is_decision(c_df.loc[i]):
              decision_positions.append( round( i /size,3) )

          if is_rationale(c_df.loc[i]):
            rationale_positions.append( round( i /size,3) )


      st.markdown("Commit Message Normalized Structure ")
      placeholder5 = st.empty()        
      with placeholder5 : st.markdown(
              """<div style='text-align:center; font-size:24px; color:gray;'>Loading... ⏳</div>""",
              unsafe_allow_html=True,
      )
      asyncio.create_task(run_async_commit_structure(placeholder5, decision_positions, rationale_positions, option)    )
      # commit_structure(decision_positions, rationale_positions)

    

    ############################ ANALYSES :   Rationale Evolution
      st.subheader("Rationale Evolution")
     
      commits_decision_density_list=[]
      for comit_id in  commits_IDs_list :
        commits_decision_density_list.append(decision_density(comit_id, df))

      commits_date_list = []
      for comit_id in  commits_IDs_list :
        rslt_df = df[df['commit ID'] == comit_id]
        date =   rslt_df.iloc[0]['commit date']
        commits_date_list.append( date )
 
      # create a dataframw with 3 comulns : comit_ID, density, author
      df_rq6 = pd.DataFrame(
          {'comit_ID': commits_IDs_list,
          'comit_date': commits_date_list,
          'comit_rationale_density': commits_rationale_density_list,
          'comit_decision_density': commits_decision_density_list
          })

      # df_rq6['Date_year'] = df_rq6['comit_date'].dt.to_period('Y')
      df_rq6['Date_year'] = [datetime.strptime(date, "%Y-%m-%d").year for date in list(df_rq6['comit_date'])]
      dff6_y = df_rq6.groupby(['Date_year']).mean(numeric_only=True)

      placeholder6 = st.empty()        
      with placeholder6 : st.markdown(
              """<div style='text-align:center; font-size:24px; color:gray;'>Loading... ⏳</div>""",
              unsafe_allow_html=True,
      )
      asyncio.create_task(run_async_rationale_evolution(placeholder6, dff6_y, option, start)    )

    # TODO: call a function for analysis
      
############## 
if __name__ == '__main__':
  st.set_page_config(page_title="Rationale Analyses Tool")
  st.title("Rationale Analyses Tool")
  # Sidebar navigation
  st.sidebar.page_link('tool.py', label='Module Analyzer')
  st.sidebar.page_link('pages/commit.py', label='Commit Message Analyzer')

  asyncio.run(main())

