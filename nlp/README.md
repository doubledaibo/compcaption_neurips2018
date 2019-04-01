This folder contains preprocessing scripts based on sentences' parse trees.
To use them, you should download stanford-corenlp-toolkit (tested with 3.8.0) and put it in this folder.
Suppose the subfolder is 'std-nlp',

to compile:

javac -cp "std-nlp/\*:./\*"  [script\_name]

to run:

java -cp "std-nlp/\*:./\*:." [script\_name] [options]

