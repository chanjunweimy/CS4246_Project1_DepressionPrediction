package noiseremoval;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;

public class NoiseRemover {
	public void removeNoise(String audioOriginal, String audioCleaned) {
		generateSampleNoise(audioOriginal);
		generateNoiseProfile();
		cleanNoiseSample(audioOriginal, audioCleaned);
	}
	
	private void generateSampleNoise(String audioOriginal) {
		Process process;
		try {
			String command = "ffmpeg -i " + audioOriginal + " -vn -ss 00:00:00 -t 00:00:01 noiseaud.wav";
			//System.out.println(command);
			process = Runtime.getRuntime().exec(command);
			InputStream is = process.getInputStream();
			InputStreamReader isr = new InputStreamReader(is);
			BufferedReader br = new BufferedReader(isr);
			String line;
			
			while ((line = br.readLine()) != null) {
				 line = line.trim();
				 if (line.startsWith("Error:")) {
					 System.out.println(line);
				 } else {
					 System.out.println(line);
				 }
			}
			br.close();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
	
	private void generateNoiseProfile() {
		Process process;
		try {
			String command = "sox noiseaud.wav -n noiseprof noise.prof";
			//System.out.println(command);
			process = Runtime.getRuntime().exec(command);
			InputStream is = process.getInputStream();
			InputStreamReader isr = new InputStreamReader(is);
			BufferedReader br = new BufferedReader(isr);
			String line;
			
			while ((line = br.readLine()) != null) {
				 line = line.trim();
				 if (line.startsWith("Error:")) {
					 System.out.println(line);
				 } else {
					 System.out.println(line);
				 }
			}
			br.close();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
	
	private void cleanNoiseSample(String audioOriginal, String audioCleaned) {
		Process process;
		try {
			String command = "sox " + audioOriginal + " " + audioCleaned + " noisered noise.prof 0.21";
			//System.out.println(command);
			process = Runtime.getRuntime().exec(command);
			InputStream is = process.getInputStream();
			InputStreamReader isr = new InputStreamReader(is);
			BufferedReader br = new BufferedReader(isr);
			String line;
			
			while ((line = br.readLine()) != null) {
				 line = line.trim();
				 if (line.startsWith("Error:")) {
					 System.out.println(line);
				 } else {
					 System.out.println(line);
				 }
			}
			br.close();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
}
