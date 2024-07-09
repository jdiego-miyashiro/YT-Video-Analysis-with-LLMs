initial_chunk_prompt_template = """
Task: Here is a partial transcript of a youtube video with timestamps. Your task is to destile the information from  the raw transcript and write a concise well structure article that summarizes the
      new or intersting information. The timestamps where the information is being extracted should be added at the end  In case part or the whole of the video is a tutorial it is important to list the steps provided.
      The reader might then go to the timestamps to check so include a chronological list or timeline summarizing the sequence of topics discussed in the video.
Part of Transcription:
{text}
PARTIAL ARTICLE:
title: Youtube video title
   
[start_timestamp---->end_timestamp]"""

refine_chunk_template = """
We are working on processing the information from a long youtube transcript and destile it into one well structure cohesive text either a Medium Article or a scientific paper whatever fits best.
The idea is that a reader with a technical backgroud can easily read our composition and understand what new information/technology/insight is being presented in the video.

We are working by parts because the transcript it's a very big.
We have made a partial document up to a certain point of the transcript: {existing_answer}

Your job is to merge and destile these partial  together into a consice, comprehensive but succint, easy to read document that explains what new insights are being showcase

Requirements for the document:
Focus on what new knowledege is the video presenting. Explain what new or intersting, technologies,tool,algorithm,discovery, are being discussed and how do they work and why are they relevant
The document should have the following format:
        Title: Youtube Video title 
        Abstract
        introduction,
        body
            paragraph for each key topic/tool/algorithm/etc.
            paragraph 1  [timestamp that supports the paragraph 1]
            paragraph 2  [timestamp that supports the paragraph 2]
            ....
            paragraph X  [timestamp that supports the paragraph X]
        conclusion
The body paragraphs should  reference the timestamps of the transcript that support what they are saying as if they were citations
Do not use bulletpoints use subtitles
The reader might then go to the timestamps to check so include a chronological list or timeline summarizing the sequence of topics discussed in the video

In case the video or part of the video is a tutorial list the steps provided.


Continue and refine the existing partial article with  more context below if appropiate in a way that is still coherent. Otherwise return the current partial article
------------
{text}
------------


PARTIALLY PROCESS DOCUMENT:"""
