package storage;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;

public class ConfigFileHandler {
	public static final String FILE_CONFIG_NAME_DEFAULT = "config.txt";
	public static final String FILE_OUTPUT_NAME_DEFAULT = "out.txt";
	public static final String FILE_OUTPUT_BIAS_NAME_DEFAULT = "out_bias.txt";
	public static final String FILEPATH_INPUT_DEFAULT = ".";
	public static final int FRAME_LENGTH_DEFAULT = 512;
	public static final double FRAME_PRE_EMPHASIS_ALPHA_DEFAULT = 0.95;
	public static final int MFCC_NUM_CEPSTRA_PER_FRAME = 13;
	public static final double MFCC_PRE_EMPHASIS_ALPHA_DEFAULT = 0;
	public static final double MFCC_LOWER_FILTER_FREQUENCY_DEFAULT = 0;
	public static final double MFCC_UPPER_FILTER_FREQUENCY_DEFAULT = 6855.4976;
	public static final int MFCC_NUMBER_MEL_FILTERS_DEFAULT = 23;
	
	public static final String[] CONFIG_CONSTANTS_FIELDS = {
		"OUTPUT_FILE_NAME",
		"OUTPUT_BIAS_FILE_NAME",
		"INPUT_FILE_PATH",
		"FRAME_LENGTH",
		"FRAME_PRE_EMPHASIS_ALPHA",
		"MFCC_NUM_CEPSTRA_PER_FRAME",
		"MFCC_PRE_EMPHASIS_ALPHA",
		"MFCC_LOWER_FILTER_FREQUENCY",
		"MFCC_UPPER_FILTER_FREQUENCY",
		"MFCC_NUMBER_MEL_FILTERS"
	};
	
	public static final String DELIMITER = ":==:";

	private String _configFileName = null;
	private String _outputFileName = null;
	private String _outputBiasFileName = null;
	private String _inputFilePath = null;
	private int _frameLength = FRAME_LENGTH_DEFAULT;
	private double _framePreEmphasisAlpha = FRAME_PRE_EMPHASIS_ALPHA_DEFAULT;
	private double _mfccPreEmphasisAlpha = MFCC_PRE_EMPHASIS_ALPHA_DEFAULT;
	private int _mfccNumCepstra = MFCC_NUM_CEPSTRA_PER_FRAME;
	private double _mfccLowerFilterFrequency = MFCC_LOWER_FILTER_FREQUENCY_DEFAULT;
	private double _mfccUpperFilterFrequency = MFCC_UPPER_FILTER_FREQUENCY_DEFAULT;
	private int _mfccNumMelFilters = MFCC_NUMBER_MEL_FILTERS_DEFAULT;
	
	private static ConfigFileHandler _config = null;
	
	public static ConfigFileHandler getInstance() {
		return getInstance(FILE_CONFIG_NAME_DEFAULT);
	}
	
	public static ConfigFileHandler getInstance(String configFileName) {
		if (_config == null) {
			_config = new ConfigFileHandler(configFileName);
		}
		return _config;
	}
	
	private ConfigFileHandler() {	
		initializeConfigFileHandle(FILE_CONFIG_NAME_DEFAULT);
	}
	
	private ConfigFileHandler(String configFileName) {
		initializeConfigFileHandle(configFileName);
	}
	
	private void initializeConfigFileHandle(String configFileName) {
		_configFileName = configFileName;
		
		File configFile = new File(configFileName);
		if (configFile.exists()) {
			readConfigFile(configFile);
		} else {
			initializeVariables();
			createConfigFile(configFile);
		}
	}
	
	public void generateConfigFile() {
		File configFile = new File(_configFileName);
		if (configFile.exists()) {
			configFile.delete();
		} 
		initializeVariables();	
		createConfigFile(configFile);
	}

	private void readConfigFile(File configFile) {

		try (
				FileReader fr = new FileReader(configFile);
	            BufferedReader br = new BufferedReader(fr);
				) {

            String line = br.readLine();
            while(line != null){
            	line = line.trim();
            	
            	if (line.isEmpty()) {
            		continue;
            	}
            	
            	String value = line.split(DELIMITER)[1];
            	if (line.startsWith(CONFIG_CONSTANTS_FIELDS[0])) {
            		_outputFileName = value;
            	} else if (line.startsWith(CONFIG_CONSTANTS_FIELDS[1])) {
            		_outputBiasFileName = value;
            	} else if (line.startsWith(CONFIG_CONSTANTS_FIELDS[2])) {
            		_inputFilePath = value;
            	} else if (line.startsWith(CONFIG_CONSTANTS_FIELDS[3])) {
            		_frameLength = Integer.parseInt(value);
            	} else if (line.startsWith(CONFIG_CONSTANTS_FIELDS[4])) {
            		_framePreEmphasisAlpha = Double.parseDouble(value);
            	} else if (line.startsWith(CONFIG_CONSTANTS_FIELDS[5])) {
            		_mfccNumCepstra = Integer.parseInt(value);
            	} else if (line.startsWith(CONFIG_CONSTANTS_FIELDS[6])) {
            		_mfccPreEmphasisAlpha = Double.parseDouble(value);
            	} else if (line.startsWith(CONFIG_CONSTANTS_FIELDS[7])) {
            		_mfccLowerFilterFrequency = Double.parseDouble(value);
            	} else if (line.startsWith(CONFIG_CONSTANTS_FIELDS[8])) {
            		_mfccUpperFilterFrequency = Double.parseDouble(value);
            	} else if (line.startsWith(CONFIG_CONSTANTS_FIELDS[9])) {
            		_mfccNumMelFilters = Integer.parseInt(value);
            	} else {
            		//open for modification
            	}
            	
            	
                line = br.readLine();
            }
            fr.close();
            br.close();
        }catch (Exception e){
            e.printStackTrace();
        }		

	}
	
	private void initializeVariables() {
		_outputFileName = FILE_OUTPUT_NAME_DEFAULT;
		_outputBiasFileName = FILE_OUTPUT_BIAS_NAME_DEFAULT;
		_inputFilePath = FILEPATH_INPUT_DEFAULT;
		_frameLength = FRAME_LENGTH_DEFAULT;
		_framePreEmphasisAlpha = FRAME_PRE_EMPHASIS_ALPHA_DEFAULT;
		_mfccPreEmphasisAlpha = MFCC_PRE_EMPHASIS_ALPHA_DEFAULT;
		_mfccNumCepstra = MFCC_NUM_CEPSTRA_PER_FRAME;
		_mfccLowerFilterFrequency = MFCC_LOWER_FILTER_FREQUENCY_DEFAULT;
		_mfccUpperFilterFrequency = MFCC_UPPER_FILTER_FREQUENCY_DEFAULT;
		_mfccNumMelFilters = MFCC_NUMBER_MEL_FILTERS_DEFAULT;
	}
	
	private void createConfigFile(File configFile) {
		final Object[] CONFIG_CONSTANTS_VALUES = {
			FILE_OUTPUT_NAME_DEFAULT,
			FILE_OUTPUT_BIAS_NAME_DEFAULT,
			FILEPATH_INPUT_DEFAULT,
			FRAME_LENGTH_DEFAULT,
			FRAME_PRE_EMPHASIS_ALPHA_DEFAULT,
			MFCC_NUM_CEPSTRA_PER_FRAME,
			MFCC_PRE_EMPHASIS_ALPHA_DEFAULT,
			MFCC_LOWER_FILTER_FREQUENCY_DEFAULT,
			MFCC_UPPER_FILTER_FREQUENCY_DEFAULT,
			MFCC_NUMBER_MEL_FILTERS_DEFAULT
		};
		
		try (FileWriter fw = new FileWriter(configFile, true)){
			
			for (int i = 0; i < CONFIG_CONSTANTS_FIELDS.length; i++) {
				fw.write(CONFIG_CONSTANTS_FIELDS[i] + DELIMITER + CONFIG_CONSTANTS_VALUES[i] + System.lineSeparator());
			}
			
			fw.close();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}	
	
	public String getOutputBiasFileName() {
		return _outputBiasFileName;
	}
	
	public String getOutputFileName() {
		return _outputFileName;
	}
	
	public String getInputFilePath() {
		return _inputFilePath;
	}

	public int getFrameLength() {
		return _frameLength;
	}

	public double getFramePreEmphasisAlpha() {
		return _framePreEmphasisAlpha;
	}
	
	public int getMfccNumCepstra() {
		return _mfccNumCepstra;
	}
	
	public double getMfccPreEmphasisAlpha() {
		return _mfccPreEmphasisAlpha;
	}
	
	public double getMfccLowerFilterFrequency() {
		return _mfccLowerFilterFrequency;
	}

	public double getMfccUpperFilterFrequency() {
		return _mfccUpperFilterFrequency;
	}
	
	public int getMfccNumMelFilters() {
		return _mfccNumMelFilters;
	}
}
