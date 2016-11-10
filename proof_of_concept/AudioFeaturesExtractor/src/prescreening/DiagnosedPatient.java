package prescreening;

public class DiagnosedPatient {
	private String _name = null;
	private double _consultationLength = 0;
	private boolean _isDepressed = false;
	private double _gpConfidence = 0;
	
	public DiagnosedPatient() {
	}

	public String getName() {
		return _name;
	}

	public void setName(String name) {
		this._name = name;
	}

	public double getConsultationLength() {
		return _consultationLength;
	}

	public void setConsultationLength(double consultationLength) {
		this._consultationLength = consultationLength;
	}

	public boolean getIsDepressed() {
		return _isDepressed;
	}

	public void setIsDepressed(boolean isDepressed) {
		this._isDepressed = isDepressed;
	}

	public double getGpConfidence() {
		return _gpConfidence;
	}

	public void setGpConfidence(double gpConfidence) {
		this._gpConfidence = gpConfidence;
	}
	
	public String getPatientInfo() {
		if (_name == null) {
			return null;
		}
		
		StringBuffer buffer = new StringBuffer();
		buffer.append(_name);
		buffer.append(": GP is ");
		buffer.append(_gpConfidence * 100);
		buffer.append("% confidence to say that the patient is ");
		if (_isDepressed) {
			buffer.append("depressed");
		} else {
			buffer.append("normal");
		}
		buffer.append(" and required ");
		buffer.append(_consultationLength);
		buffer.append("h.");
		return buffer.toString();
	}

}
