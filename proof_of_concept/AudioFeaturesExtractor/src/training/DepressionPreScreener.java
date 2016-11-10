package training;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;

import prescreening.DiagnosedPatient;

public class DepressionPreScreener {
	public static final String FILEPATH_PRESCREEN = "data/dcapswoz_audio/dev";
	public static final String FILEPATH_PATIENTS = "names.txt";
	public static final double NORMAL_CONFIDENCE_THRESHOLD = 0.7;

	
	private DepressionPreScreener() {
	}
	
	public void execute(String[] args) {
		ArrayList<DiagnosedPatient> patients = extractPatientDatas(args);
		patients = classifyDepression(patients);
		showPatientsInfo(patients);
		patients = rejectNormalPerson(patients);
		
	}

	private ArrayList<DiagnosedPatient> rejectNormalPerson(ArrayList<DiagnosedPatient> patients) {
		for (DiagnosedPatient patient : patients) {
			if (!patient.getIsDepressed() && patient.getGpConfidence() - NORMAL_CONFIDENCE_THRESHOLD > 0) {
				patients.remove(patient);
			}
		}
		return patients;
	}

	private void showPatientsInfo(ArrayList<DiagnosedPatient> patients) {
		for (DiagnosedPatient patient : patients) {
			System.out.println(patient.getPatientInfo());
		}
	}

	private ArrayList<DiagnosedPatient> classifyDepression(ArrayList<DiagnosedPatient> patients) {
		Process process;
		try {
			String command = "python onlinegp.py " + AudioFeaturesGenerator.EMOTION_DCAPSWOZ_ALL;
			//System.out.println(command);
			process = Runtime.getRuntime().exec(command);
			InputStream is = process.getInputStream();
			InputStreamReader isr = new InputStreamReader(is);
			BufferedReader br = new BufferedReader(isr);
			String line;
			
			int i = 0;
			while ((line = br.readLine()) != null) {
				  String[] tokens = line.split(" ");
				  int depressedNum = Integer.parseInt(tokens[0]);
				  boolean isDepressed = depressedNum == 1;
				  patients.get(i).setIsDepressed(isDepressed);
				  patients.get(i).setGpConfidence(Double.parseDouble(tokens[depressedNum + 1])); 
				  i++;
			}
			br.close();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		return patients;
	}

	private ArrayList<DiagnosedPatient> extractPatientDatas(String[] args) {
		AudioFeaturesGenerator featureGenerator = new AudioFeaturesGenerator();
		if (args.length == 2) {
			trainDcapswozEmotion(featureGenerator, args[0]);
			return retrieveNameAndTimeLength(args[1]);
		} else if (args.length == 0){
			trainDcapswozEmotion(featureGenerator, FILEPATH_PRESCREEN);
			return retrieveNameAndTimeLength(FILEPATH_PATIENTS);
			//System.out.println(FILEPATH_PRESCREEN);
		}
		return null;
	}
	
	private ArrayList<DiagnosedPatient> retrieveNameAndTimeLength(String filename) {
		ArrayList<DiagnosedPatient> patients = new ArrayList<DiagnosedPatient>();
		try {
			BufferedReader br = new BufferedReader(new FileReader(filename));
			String line;
			while ((line = br.readLine()) != null) {
				String[] tokens = line.split(" ");
				String name = tokens[0];
				double timeLength = Double.parseDouble(tokens[1]);
				DiagnosedPatient patient = new DiagnosedPatient();
				patient.setName(name);
				patient.setConsultationLength(timeLength);
				patients.add(patient);
			}
			br.close();
			return patients;
		} catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		return null;
	}

	private static void trainDcapswozEmotion(AudioFeaturesGenerator featureGenerator, String trainFilePath) {
		File emotionTrain = new File(trainFilePath);
		File[] emotionFiles = emotionTrain.listFiles();
		
		File emotionAllFeaturesFile = featureGenerator.createFile(AudioFeaturesGenerator.EMOTION_DCAPSWOZ_ALL);
		
		if (!featureGenerator.computeMFCC(emotionFiles,
			    emotionAllFeaturesFile.getAbsolutePath())) {
			System.exit(-1);
		}	
	}
	
	public static void main(String[] args) {
		DepressionPreScreener prog = new DepressionPreScreener();
		prog.execute(args);
	}


}
