import streamlit as st

#Component Pkgs
import streamlit.components.v1 as components 

import base64

#Pkgs - Files
import Html
import Css
import imagecontrast, CV_face, CV_smile, CV_eyes, CV_cannize, CV_cartoonize, CV_object, CV_Pose, CV_style, CV_semantic
import DS_pandas_profiling, DS_EDA_App
import NLP_NER, NLP_pos, NLP_sentiment,NLP_qa
import SG_gender

def main():
	st.set_option('deprecation.showPyplotGlobalUse', False)
	"""STREAM AI - A Hub for NLP Apps, Computer Vision Apps, Data Science Apps, ML Apps, DL Apps, """
	Css.local_css("style.css")
	Css.side_background()

	choice = st.sidebar.selectbox("Select Activity", ["Home", "Applications", "Blogs", "About"])

	if choice == 'Home':
		pass


	elif choice == 'Applications':
		categories = ["Natural Language Processing","Computer Vision", "Speech Processing", "Data Visualization"]
		category_choice = st.sidebar.radio("Pick a Domain",categories)

		if category_choice == "Natural Language Processing":
			apps_nlp = st.sidebar.radio("Algorithm",('Named Entity Recognition', 'Part-of-Speech Tagging', 'Sentiment Detection', 'Question Answering'))
			if apps_nlp == 'Named Entity Recognition':
				NLP_NER.main()

			elif apps_nlp == 'Part-of-Speech Tagging':
				NLP_pos.main()

			elif apps_nlp == 'Sentiment Detection':
				NLP_sentiment.add_flair()

			elif apps_nlp == 'Question Answering':
				
				#st.success("NLP: Question Answering App")
				NLP_qa.add_qa()

		elif category_choice == 'Computer Vision':
			apps_cv = st.sidebar.radio("Algorithm",['Image Contrasting','Face Detection', 'Eyes Detection','Smile Detection','Cannize','Cartoonize','Style Detection','Pose Detection', 'Semantic Segmentation', 'Object Detection'])
			if apps_cv == 'Image Contrasting':
				imagecontrast.main()

			if apps_cv == 'Face Detection':
				CV_face.face_main()

			elif apps_cv == 'Eyes Detection':
				CV_eyes.eyes_main()

			elif apps_cv == 'Smile Detection':
				CV_smile.smiles_main()

			elif apps_cv == 'Style Detection':
				CV_style.detect_style()

			elif apps_cv == 'Cannize':
				CV_cannize.cannize_main()

			elif apps_cv == 'Cartoonize':
				CV_cartoonize.cartoonize_main()

			elif apps_cv == 'Pose Detection':
				CV_Pose.main()

			elif apps_cv == 'Semantic Segmentation':
				CV_semantic.semantic_segmentation()

			elif apps_cv == 'Object Detection':
				CV_object.object_main()



		elif category_choice == 'Speech Processing':
			apps_sp = st.sidebar.radio("Algorithm",['Voice Based Gender Detection'])
			if apps_sp == 'Voice Based Gender Detection':
				# st.success("SP: Voice Based Gender Detection App")
				SG_gender.add_gender()


		elif category_choice == 'Data Visualization':
			apps_ds = st.sidebar.radio("Algorithm",['DataSet Explorer', 'EDA of Iris DataSet'])
			if apps_ds == 'DataSet Explorer':
				DS_pandas_profiling.main()
			elif apps_ds == 'EDA of Iris DataSet':
				DS_EDA_App.main()



	elif choice == 'Blogs':
		domain = st.sidebar.radio("Select the Domain", ["Natural Language Processing","Computer Vision", "Speech Processing", "Data Visualization"])
		if domain == "Natural Language Processing":
			pass
		elif domain == "Computer Vision":
			components.html(Html.blog_cv(), height=2430)
		elif domain == "Speech Processing":
			pass
		elif domain == "Data Visualization":
			pass



	elif choice == 'About':
		Html.html_heading()
		st.markdown(Html.html_heading(), unsafe_allow_html = True)
		components.html(Html.html_about(), height=1000)


if __name__ == '__main__':
	main()
