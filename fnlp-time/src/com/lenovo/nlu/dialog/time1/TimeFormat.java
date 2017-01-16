/*
 * (C) Copyright LENOVO Corp. 1984-2016 - All Rights Reserved
 *
 *   The original version of this source code and documentation is copyrighted
 * and owned by LENOVO, Inc., a wholly-owned subsidiary of LENOVO. These
 * materials are provided under terms of a License Agreement between LENOVO
 * and Sun. This technology is protected by multiple US and International
 * patents. This notice and attribution to LENOVO may not be removed.
 *   LENOVO is a registered trademark of LENOVO, Inc.
 *
 */
package com.lenovo.nlu.dialog.time1;

import java.util.List;

public class TimeFormat {
	private String startTime;
	private String endTime;
    private List<String> times;

	public TimeFormat(String startTime, String endTime, List<String> times) {
		super();
		this.startTime = startTime;
		this.endTime = endTime;
	}

	public String getStartTime() {
		return startTime;
	}

	public void setStartTime(String startTime) {
		this.startTime = startTime;
	}

	public String getEndTime() {
		return endTime;
	}

	public void setEndTime(String endTime) {
		this.endTime = endTime;
	}

	public List<String> getTimes() {
		return times;
	}

	public void setTimes(List<String> times) {
		this.times = times;
	}
	@Override
	public String toString() {
		return "TimeFormat: [startTime=" + startTime + ", endTime=" + endTime
				+ "], " + times.toString();
	}
	public String toString2() {
		return "TimeFormat: [startTime=" + startTime + ", endTime=" + endTime
				+ "]" ;
	}
}
