import streamlit as st
from wordcloud import WordCloud,STOPWORDS
import  matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import seaborn as sns
import numpy as np
import asyncio
import pandas as pd
import re
import requests
from tensorflow.keras.models import load_model



# Define a function to fetch commits from paginated GitHub API
def fetch_commits(url, username, classic_token):
    commit_messages= []
    while url:
        response = requests.get(url, auth=( username , classic_token ))
        if response.status_code == 200:
            data = response.json()
            # Append commit messages to the list
            for commit in data:
                commit_messages.append(commit)
            
            # Check if there is a next page of results
            if 'Link' in response.headers:
                links = response.headers['Link']
                # Parse the 'Link' header to find the next URL
                if 'rel="next"' in links:
                    next_url = links.split(",")[0].split(";")[0].strip("<> ")
                    url = next_url
                else:
                    url = None
            else:
                url = None
        else:
            raise Exception(f"Failed to fetch data: {response.status_code}")

    return commit_messages


def is_source_code(txt):
        # returns ttRue if source code
      #  if txt.startswith(" ") and txt.endswith(";"): 
       #     return True

        #if txt.startswith("\t") and txt.endswith(";"): 
         #   return True
        
        if txt.endswith(";"): 
            return True
        
        if "$ echo" in txt or "# echo" in txt or "$ cd" in txt or "$ cat" in txt:
            return True

        if txt.startswith(" git") or txt.startswith("# grep") or txt.startswith("  git grep"): 
            return True
        
        if txt.startswith("bash") or txt.startswith("grep") or txt.startswith("ls"): 
            return True
        
        if txt.replace("\t","").startswith("#include <") : 
            return True
        
        if txt.replace("\t","").startswith("#define ") : 
            return True
        
        if txt.startswith("spatch ") : 
            return True
        
        # if a line is only composed of non-alphabetic letters
        if  ( re.sub('[a-zA-Z]+', '', txt) == txt ) :
            return True

        
        if "(" in txt and ")" in txt and "{" in txt:
            return True
        
        if "if" in txt and "(" in txt and txt.endswith(")"):
            return True
        
        if  txt.replace(" ","").startswith("for") and "(" in txt and txt.endswith(")"):
            return True
        
        if txt.replace(" ","").startswith("/*") or txt.replace(" ","").endswith("*/"):
            return True
        
        if txt.replace(" ","").startswith("*"):
            return True

        if txt.replace(" ","").startswith("case") and txt.replace(" ","").endswith(":"):
            return True
        
        return False


def preprocess(row):
    message = row['message']

    index = message.find("Link: ")
    if index > 0 : 
        message = message[:index]

    index2 = message.find("Signed-off-by:")
    if index2 > 0 : 
        message = message[:index2]

    index2 = message.find("Signed-Off-by:")
    if index2 > 0 : 
        message = message[:index2]

    index2 = message.find("Signed-Off-By:")
    if index2 > 0 : 
        message = message[:index2]


    index3 = message.find("Cc:")
    if index3 > 0 : 
        message = message[:index3]

    index4 = message.find("Suggested-by:")
    if index4 > 0 : 
        message = message[:index4]

    index4 = message.find("Requested-by:")
    if index4 > 0 : 
        message = message[:index4]

    index4 = message.find("Debbuged-by:")
    if index4 > 0 : 
        message = message[:index4]

    index4 = message.find("Acked-by:")
    if index4 > 0 : 
        message = message[:index4]

    index4 = message.find("Reported-by:")
    if index4 > 0 : 
        message = message[:index4]
    
    index4 = message.find("Reviewed-by:")
    if index4 > 0 : 
        message = message[:index4]

    index4 = message.find("Tested-by:")
    if index4 > 0 : 
        message = message[:index4]

        # Regex expression for "["" then "email adress" then ":" then "space" then "any chacrcter except ]" then "]".
    x = re.search("\[([A-Za-z0-9]+[.-_])*[A-Za-z0-9,-]+@[A-Za-z0-9-]+(\.[A-Z|a-z]{2,})+:\s[^]]*\]", message)
    if x:
        (a,b) = x.span() 
        message = message[:a] 

    # remove refrences formatted like this: \n [\d]space any charcter except \n, then \n,                                   TODO:  fix commit 203 
    x = re.search("\n\[\d\]\s.*\n", message)
    if x:
        (a,b) = x.span() 
        message = message[:a] 

    #  remove URLS , exp; commit 84 
    x = re.findall("http[^\s]*\s", message)
    for s in x:
  	    message=message.replace(s,"")

    #  remove stack call like this [ .  ] text \n , exp; commit 61
    x = re.findall("\n\[[\s]*[\d]+\.[\d]+\]\s.*", message)
    for s in x:
  	    message=message.replace(s,"")
    

    x = re.findall("\n\[[\s]*[\d]+\.[\d]+\]\[.*\]\s.*", message) # commit 55 
    for s in x:
  	    message=message.replace(s,"")
            

    # remove  call traces , exp: commit  82, 106 , 48, 
    x = re.search("Call Trace:\n([^\n]+\n)+\n", message)
    if x:
        (a,b) = x.span() 
        message = message[:a]  + message[b:] 


    # split to sentences, if a sentence starts with space and ends with; remove it ; if just 3 character remove line
    sentences = message.split('\n')
    title = sentences[0]

    non_code_sentences = [s for s in sentences[1:] if not is_source_code(s) and len( s.replace(" ", "") ) > 3 ] # only the body
    non_code_message = "\n".join(non_code_sentences)
    
    return [title] + re.split("\.\s|\:\n",non_code_message)


def get_commit(commitID, df_current):
  # selecting rows of the commit
  c_df = df_current[df_current['commit ID'] == commitID]
  return c_df


# to be applied by sentence
def is_rationale(row):
  if (  row["Rationale"] == 1):
    return True
  return False


# to be applied by sentence
def is_decision(row):
  if (  row["Decision"] == 1):
    return True
  return False


def load_models():
    model_rationale = load_model("models/bi_lstm_model_rationale.h5")
    model_decision = load_model("models/bi_lstm_model_decision.h5")
    return model_rationale, model_decision
   

def commit_contains_rationale(commitID,  df_current):
  # selecting rows of the commit
  rslt_df = df_current[df_current['commit ID'] == commitID]
  #print(rslt_df)

  # select rows of the commit that contain rationale
  df2 = rslt_df.apply(lambda x : True
            if is_rationale(x) else False, axis = 1)
  #print(df2)

  # at least one sentence is labeled as rationale
  #print(len(df2[df2 == True]))
  if len(df2[df2 == True].index) > 0 :
    return True

  return False


def rationale_density(commitID, df_current):
  # selecting rows of the commit
  c_df = df_current[df_current['commit ID'] == commitID]
 # print(c_df)

  # select rows of the commit that contain rationale
  df2 = c_df.apply(lambda x : True
            if is_rationale(x) else False, axis = 1)
  #print(df2)

  return len(df2[df2 == True].index) / len(c_df)


def decision_density(commitID, df_current):
  # selecting rows of the commit
  c_df = df_current[df_current['commit ID'] == commitID]
 # print(c_df)

  # select rows of the commit that contain decision
  df2 = c_df.apply(lambda x : True
            if is_decision(x) else False, axis = 1)
  #print(df2)

  return len(df2[df2 == True].index) / len(c_df)


def commit_size(commitID, df_current):
  # selecting rows of the commit
  c_df = df_current[df_current['commit ID'] == commitID]
  return len(c_df)


def number_of_commits_with_size_and_rationale_density(si,ri, commits_IDs_list, df_current):
  n = 0
  for comit_id in  commits_IDs_list :
    if commit_size(comit_id, df_current) == si and rationale_density(comit_id, df_current) ==ri:
      n = n +1
  return n


def distribution_categories(df_current):

    decision_only_sentences = list( df_current.loc[ (df_current['Decision'] == 1) & (df_current['Rationale'] == 0)  ]['message_preprocessed'])
    rationale_only_sentences = list( df_current.loc[ (df_current['Decision'] == 0) & (df_current['Rationale'] == 1)  ]['message_preprocessed'])
    decision_rationale_sentences = list( df_current.loc[ (df_current['Decision'] == 1) & (df_current['Rationale'] == 1)  ]['message_preprocessed'])
    no_no_sentences = list( df_current.loc[ (df_current['Decision'] == 0) & (df_current['Rationale'] == 0) ]['message_preprocessed'])

    st.markdown( "Decision only sentences: " + str( len(decision_only_sentences)))
    st.markdown("Rationale only sentences: " + str(  len(rationale_only_sentences)))
    st.markdown("Decision & Rationale sentences: " + str( len(decision_rationale_sentences)) )
    st.markdown("No Decision and No Rationale sentences: " + str( len(no_no_sentences)))

    return decision_only_sentences, rationale_only_sentences, decision_rationale_sentences, no_no_sentences 


async def word_clouds(df_current, text_list, system_name, L):
    more_stopwords = set({"kernel", "linux", "task", "thread", "process", "system", "patch"})
    fig = plt.figure(figsize=(4,4))
    cloud_im = WordCloud(
                          stopwords=STOPWORDS.union(more_stopwords),
                          background_color='white',
                          collocations=False,
                          width=2500,
                          height=1500
                         ).generate(" ".join(text_list))

    #plt.axis('off')  # this should work for a pdf
    plt.xticks([], [])  # a workaround to have a frame with an SVG
    plt.yticks([], [])
    plt.imshow(cloud_im,  interpolation="bilinear")
    plt.savefig('figures\word_clouds_'+ system_name + "_" +  L+ '_.pdf', bbox_inches='tight')
    return fig


@st.cache_data
def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv().encode("utf-8")


async def factors_commit_size(df_current, commits_length_list, commits_rationale_density_list, commits_IDs_list ):
  sns.set_style('white')

  title_font_size = 24
  text_font_size = 10
  tick_label_size = 8
  legend_font_size = 8
  annotation_fontsize = 6

  marker_sizes = [12*int(number_of_commits_with_size_and_rationale_density(si, commits_rationale_density_list[i],commits_IDs_list, df_current )) for i, si in enumerate(commits_length_list)]
  marker_colors = [int(number_of_commits_with_size_and_rationale_density(si, commits_rationale_density_list[i], commits_IDs_list, df_current)) for i, si in enumerate(commits_length_list)]

  fig = plt.figure(figsize=(3.8,4.3))
  sc1 = plt.scatter(
        commits_length_list, commits_rationale_density_list,
        s=marker_sizes,
        c=marker_colors,
        # color='orange',
        cmap="Spectral",# "RdYlGn", #"seismic",# "Spectral",
        # alpha=0.7
        )#, s=marker_sizes)
  plt.xlabel("Commit message size (Number of sentences)", fontsize=text_font_size)
  plt.ylabel("Rationale Density", fontsize=text_font_size)
    #plt.title = ('Rationale Density ...')
    # We change the fontsize of minor ticks label
  plt.tick_params(axis='both', which='major', labelsize=tick_label_size)
  plt.tick_params(axis='both', which='minor', labelsize=tick_label_size)

  plt.legend(*sc1.legend_elements(),
          title="Number of commits", loc='upper right',
           ncol=2,fancybox=True,
           frameon=True,
           framealpha=0.5,
           fontsize=legend_font_size
           )#, shadow=True)

  plt.tight_layout()
  plt.show() 
  #st.pyplot(fig) 
  return fig


def number_of_authors_with_number_of_commits_and_rationale_density(ci,ri, dff4):
  n = 0
  for author in  list(dff4.index) :
    c= int(dff4.loc[dff4.index == author]['Number of commits'])
    r = float(dff4.loc[dff4.index == author]['comit_rationale_density'])
    if ( c == ci ) & ( r == ri) :
      n = n +1
  return n


async def commit_density(user_commit):
  return 0.15


async def factors_developers(dff4, commits_length_list, commits_rationale_density_list, commits_IDs_list , x,y):
  
  title_font_size = 24
  text_font_size = 10
  tick_label_size = 8
  legend_font_size = 8
  annotation_fontsize = 6

  marker_sizes = [
    20*int(number_of_authors_with_number_of_commits_and_rationale_density(list(dff4['Number of commits'])[i],list(dff4['comit_rationale_density'])[i], dff4))\
    for i, fi in enumerate(list(dff4.index))
    ]
  marker_colors = [
    int(number_of_authors_with_number_of_commits_and_rationale_density(list(dff4['Number of commits'])[i],list(dff4['comit_rationale_density'])[i], dff4))\
    for i, fi in enumerate(list(dff4.index))
    ]

  fig = plt.figure(figsize=(3.8,4.3))
  sc1 = plt.scatter(
      x, y,
      s=marker_sizes,
      c=marker_colors,
      # color='orange',
      cmap= "RdYlGn",#"Spectral",
      )#, s=marker_sizes)

  plt.xlabel("Number of commits per author", fontsize=text_font_size)
  plt.ylabel("Average Rationale Density", fontsize=text_font_size)

  # We change the fontsize of minor ticks label
  plt.tick_params(axis='both', which='major', labelsize=tick_label_size)
  plt.tick_params(axis='both', which='minor', labelsize=tick_label_size)

  # only one line may be specified; ymin & ymax specified as a percentage of y-range
  axvline1 = plt.axvline(x=16, ymin=0.5, ymax=0.8, color='purple', ls='--')

  axhline2 = plt.hlines(y=0.2, xmin=0, xmax=15, color='teal', ls='--')
  axhline3 = plt.hlines(y=0.7, xmin=0, xmax=15, color='green', ls='--')

  legend1 = plt.legend([axvline1, axhline2, axhline3],['Number of commits = 16', 'Rationale density = 0.2', 'Rationale density = 0.7'], loc='lower right')

  plt.legend(*sc1.legend_elements(),
            title="Number of authors", loc='upper right',
            ncol=2,fancybox=True,
            frameon=True,
            framealpha=0.5,
            fontsize=legend_font_size
            )#, shadow=True)

  plt.gca().add_artist(legend1)

  plt.tight_layout()
  plt.show()
  return fig
  #st.pyplot(fig) 


async def commit_structure(decision_positions, rationale_positions, system_name):
  fig, ax = plt.subplots(figsize=(5.2,3))
  bins = np.arange(0,1.1,0.1)
  
  ax.hist([decision_positions,rationale_positions],label=['Decision','Rationale'], color = ["#ADD8E6","#F4C2C2"], edgecolor='k',bins=bins)
  plt.xticks(bins)

  plt.xlabel('Normalized position of the sentence in the commit message')
  plt.ylabel('Number of commits')

  plt.legend(framealpha=0)
  plt.savefig('figures\commit_strcuture_normalized_'+ system_name +'.pdf', bbox_inches='tight')
  plt.show()
  return fig


async def rationale_evolution(dff6_y, system_name):
  fig, ax = plt.subplots(figsize=(12,3))
  text_font_size = 10

  dff6_y['comit_decision_density'].plot(ax=ax, color='#ADD8E6', marker='o',alpha=1,linewidth=2,path_effects=[pe.Stroke(linewidth=2, foreground='k'), pe.Normal()])
  dff6_y['comit_rationale_density'].plot(ax=ax, color='#F4C2C2', marker='*', alpha=1, linewidth=2, path_effects=[pe.Stroke(linewidth=2, foreground='k'), pe.Normal()])

  ax.set_xlabel("Time (years)", fontsize=text_font_size)
  ax.set_ylabel("Average Density", fontsize=text_font_size)
  fig.legend(['Average decision density','Average rationale density'], loc='lower center', bbox_to_anchor=(0.68, 0.15), framealpha=0)

  plt.savefig('figures\evolution_decision_rationale_'+ system_name +'.pdf', bbox_inches='tight')
  plt.show()
  return fig