/*
 extract nouns and other words from gt sentences
*/
import org.json.simple.JSONArray;
import org.json.simple.JSONObject;
import org.json.simple.parser.JSONParser;
import org.json.simple.parser.ParseException;

import java.util.ArrayList;
import java.util.List;
import java.util.TreeSet;
import java.util.Iterator;
 
import java.io.FileReader;
import java.io.FileWriter;
import java.io.StringReader;

import edu.stanford.nlp.ling.HasWord;
import edu.stanford.nlp.ling.TaggedWord;
import edu.stanford.nlp.parser.lexparser.LexicalizedParser;
//import edu.stanford.nlp.parser.shiftreduce.ShiftReduceParser;
import edu.stanford.nlp.process.DocumentPreprocessor;
import edu.stanford.nlp.tagger.maxent.MaxentTagger;
import edu.stanford.nlp.trees.Tree;

public class ExtractWordPos {

	public static void main(String[] args) throws Exception {
		int curArg = 0;
		String fin = "";
		String fout = "";
		boolean pos = false;
		String modelPath = "edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz";
		//String modelPath = "edu/stanford/nlp/models/srparser/englishSR.beam.ser.gz";
		String taggerPath = "edu/stanford/nlp/models/pos-tagger/english-bidirectional/english-bidirectional-distsim.tagger";
		while (curArg < args.length) {
			switch (args[curArg]) {
				case "-fin":
					fin = args[curArg + 1];
					curArg += 2;
					break;
				case "-fout":
					fout = args[curArg + 1];
					curArg += 2;
					break;
				default:
					System.err.println("Unknown option");
					System.exit(1);
			} 
		}
		ArrayList<Long> ids = new ArrayList<Long>();
		ArrayList<String> sentences = new ArrayList<String>();
		JSONParser json = new JSONParser();
		JSONArray input;
		input = (JSONArray) json.parse(new FileReader(fin));
		for (Object o : input) {
			JSONObject item = (JSONObject) o;
			ids.add((Long)item.get("id"));
			sentences.add((String) item.get("raw"));	
		}

		MaxentTagger tagger = new MaxentTagger(taggerPath);
		LexicalizedParser model = LexicalizedParser.loadModel(modelPath);
		//ShiftReduceParser model = ShiftReduceParser.loadModel(modelPath);
		ExtractWordPos collector = new ExtractWordPos();
		TreeSet<String> nouns = new TreeSet<String>();
		TreeSet<String> others = new TreeSet<String>();	
		for (int idx = 0; idx < sentences.size(); ++idx) {
			if (idx % 100 == 0)
				System.err.println(idx + " / " + sentences.size());
			String sentence = sentences.get(idx);
			long id = ids.get(idx);	
			DocumentPreprocessor tokenizer = new DocumentPreprocessor(new StringReader(sentence));
			for (List<HasWord> tokenized_sentence : tokenizer) {
				List<TaggedWord> tagged = tagger.tagSentence(tokenized_sentence);
				for (TaggedWord word : tagged) {
					String tag = word.tag();
					if (tag.startsWith("NN")) {
						String aaa = word.word();
						nouns.add(aaa); }
					else
						others.add(word.word());
				}
				break;
			}
		}
		JSONObject words = new JSONObject();
		JSONArray nounlist = new JSONArray();
		Iterator itr = nouns.iterator();
		while (itr.hasNext()) {
			String word = (String) itr.next();
			nounlist.add(word);
		}
		JSONArray otherlist = new JSONArray();
		itr = others.iterator();
		while (itr.hasNext()) {
			String word = (String) itr.next();
			otherlist.add(word);
		}
		words.put("nouns", nounlist);
		words.put("others", otherlist);
		FileWriter file = new FileWriter(fout);
		file.write(words.toJSONString());
		file.flush();
		
	}	

}
