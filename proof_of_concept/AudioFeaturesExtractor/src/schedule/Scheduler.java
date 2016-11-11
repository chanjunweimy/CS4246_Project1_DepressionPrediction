package schedule;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.time.DayOfWeek;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.ArrayList;

import prescreening.DiagnosedPatient;

public class Scheduler {
	public static final String FILE_COUNSELLOR = "counsellor_available.txt";
	public static final String FILE_DAY = "day_config.txt";
	
	private ArrayList<Counsellor> _counsellors = new ArrayList<Counsellor>();
	
	public Scheduler() {
		initializeScheduler();
	}
	
	public void runScheduler(ArrayList<DiagnosedPatient> patients) {
		int size = _counsellors.get(0).getTimeSlots().size();
		for (DiagnosedPatient patient:patients) {
			long minutes = (long) (patient.getConsultationLength() * 60);
			boolean isDone = false;
			for (int i = 0; i < size; i++) {
				for (int j = 0; j < _counsellors.size(); j++) {
					TimeSlot slot = _counsellors.get(j).getTimeSlots().get(j);
					if (slot.isFreeSlot() && !slot.isTooShort(minutes)) {
						LocalDateTime endTime = slot.getStart().plusMinutes(minutes);
						TimeSlot newSlot = new TimeSlot();
						newSlot.setStart(slot.getStart());
						newSlot.setEnd(endTime);
						newSlot.setPatient(patient);
						_counsellors.get(j).addTimeSlot(newSlot);
						_counsellors.get(j).getTimeSlots().get(i).setStart(endTime);
						isDone = true;
						break;
					}
				}
				if (isDone) {
					break;
				}
			}
		}
		
		for (Counsellor counsellor: _counsellors) {
			counsellor.printCounsellor();
		}
	}
	
	private void initializeScheduler() {
		initializeCounsellors();
		addTimeSlots();
		for (Counsellor counsellor: _counsellors) {
			counsellor.printCounsellor();
		}
	}

	private void addTimeSlots() {
		LocalDateTime mon = getNearestMonday();
		
		
		ArrayList<String> startStrings = new ArrayList<String>();
		ArrayList<String> endStrings = new ArrayList<String>();
		try {
			BufferedReader br = new BufferedReader(new FileReader(FILE_DAY));
			String line;
			while ((line = br.readLine()) != null) {
				line = line.trim().toLowerCase();
				if (line.isEmpty()) {
					continue;
				}
				String[] tokens = line.split("-");
				startStrings.add(tokens[0]);
				endStrings.add(tokens[1]);
			}
			br.close();
		} catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
		
		DateTimeFormatter formatter = DateTimeFormatter.ofPattern("yyyy-MM-dd");
		DateTimeFormatter dateTimeformatter = DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm");

		for (int i = 0; i < 5; i ++) {
			LocalDateTime day = mon.plusDays(i);
			String text = day.format(formatter);
			
			for (int j = 0; j < startStrings.size(); j++) {
				for (int k = 0; k < _counsellors.size(); k++) {
					LocalDateTime start = LocalDateTime.parse(text + " " + startStrings.get(j), dateTimeformatter);
					LocalDateTime end = LocalDateTime.parse(text + " " + endStrings.get(j), dateTimeformatter);
					
					TimeSlot slot = new TimeSlot();
					slot.setStart(start);
					slot.setEnd(end);
					_counsellors.get(k).addTimeSlot(slot);
				}
			}	
		}
		
	}

	private LocalDateTime getNearestMonday() {
		LocalDateTime now = LocalDateTime.now();
		while (now.getDayOfWeek() != DayOfWeek.MONDAY) {
			now = now.plusDays(1);
		}
		return now;
	}

	private void initializeCounsellors() {
		try {
			BufferedReader br = new BufferedReader(new FileReader(FILE_COUNSELLOR));
			String line;
			while ((line = br.readLine()) != null) {
				line = line.trim().toLowerCase();
				if (line.isEmpty()) {
					continue;
				}
				Counsellor counsellor = new Counsellor();
				counsellor.setName(line);
				_counsellors.add(counsellor);
			}
			br.close();
		} catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
	
	public static void main(String[] args) {
		new Scheduler();
	}
}
