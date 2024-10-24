import streamlit as st
from helpers import *
import asyncio
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences


def  user_rationale_density(df_current):
  df2 = df_current.apply(lambda x : True
            if is_rationale(x) else False, axis = 1)
  return len(df2[df2 == True].index) / len(df_current)


def user_decision_density(df_current):
  df2 = df_current.apply(lambda x : True
            if is_decision(x) else False, axis = 1)
  return len(df2[df2 == True].index) / len(df_current)


############### Page
async def user_commit_page():
    st.header("Commit Message Rationale Analyses")
    
    sample_commit = """mm: make function get_attribute() public.

     I changed the visibility of function get_attribute(). 
     Now, bug number #333 is fixed.
     Debbuged-by: Mouna
     """
    user_commit = st.text_area("Enter your commit message",sample_commit, height = 200)
    submit = st.button('Start Commit Analysis')  
    if submit: 
        with st.spinner('Preprocessing Your Commit Message'):
            df_user = pd.DataFrame(columns=['message', 'Decision', 'Rationale'])

            user_row = {'message': user_commit, 'Decision': 0, 'Rationale': 0}
            df_user = pd.concat([df_user, pd.DataFrame([user_row])], ignore_index=True)

            df_user['message_preprocessed'] = df_user.apply(preprocess, axis=1)
            df_user = df_user.explode('message_preprocessed')
        
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
            X_test =  df_user['message_preprocessed']

        with st.spinner('Classification - Applying classifiers'):
        ###### apply the calssifiers
            # Convert text to sequences
            sequences = tokenizer.texts_to_sequences(X_test.astype(str))
            padded_sequences = pad_sequences(sequences, maxlen=max_sequence_length)
            # Making Decision predictions
            y_pred_prob = model_decision.predict(padded_sequences) 
            y_pred = (y_pred_prob > 0.5).astype(int)  # Threshold probabilities to get binary predictions
            y_pred = y_pred.flatten()
            
            df_predicted = pd.DataFrame({
                                        'message_preprocessed': X_test, 
                                        'Decision':y_pred
                                        })

            # Making Rationale predictions
            y_pred_prob = model_rationale.predict(padded_sequences) 
            y_pred = (y_pred_prob > 0.5).astype(int)  # Threshold probabilities to get binary predictions
            y_pred = y_pred.flatten()
            df_predicted['Rationale'] = y_pred

    
        ################## Visualizae
            st.write(df_predicted)
            st.write( "Number of sentences: " + str(len(df_predicted)))
            user_commit_rationale_density = user_rationale_density(df_predicted)
            user_commit_decision_density = user_decision_density(df_predicted)
            st.write( f"Your commit message's rationale density: {user_commit_rationale_density:.2%}")
            st.write( f"Your commit message's decision density: {user_commit_decision_density:.2%}")

            if user_commit_rationale_density < 0.5 : 
                st.error('Low Rationale! Please justify more explicitly your changes', icon="🚨")
            else:
                st.success('High Rationale! Your commit message is well justified!', icon="✅")



##############
if __name__ == '__main__':
    st.set_page_config(page_title="Linux Rationale Analyses Tool")
    st.title("Linux Rationale Analyses Tool")
    # Sidebar navigation
    st.sidebar.page_link('tool.py', label='Module Analyzer')
    st.sidebar.page_link('pages/commit.py', label='Commit Message Analyzer')
    
    asyncio.run(user_commit_page())