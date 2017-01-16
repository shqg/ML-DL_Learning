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
package com.lenovo.nlu.dialog.time.holiday;

import java.io.IOException;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Calendar;
import java.util.HashMap;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * @author zhoudan,ruixiang HAO
 * 
 */
public class StandardTime {
	private HashMap<String, String> holidayMap;
	private String[] holidays = TimeTerm.HOLIDAY;
	private static SimpleDateFormat sdf = new SimpleDateFormat("yyyy年MM月dd日");// 时间格式

	public StandardTime() {
		holidayMap = createMap();
		Arrays.sort(holidays);
		initHmap();// 初始化日期
	}

	/**
	 * 初始化节日日期对应表
	 */
	private void initHmap() {
		Calendar cal = Calendar.getInstance();
		int year = cal.get(Calendar.YEAR);
		TimeUtils lauar = new TimeUtils(year, 1, 1);
		int size = holidays.length;
		while (size > 0) {
			String lh = lauar.getLunarHoliday();// 获取农历节日
			String h = lauar.getHoliday();// 获取公历节日
			String s = lauar.getSoralTerm();// 获取节气
			if (holidayMap.containsKey(lh)) {
				String value = sdf.format(lauar.getCalendar().getTime());
				holidayMap.put(lh, value);
				size--;
			}

			if (holidayMap.containsKey(h)) {
				String value = sdf.format(lauar.getCalendar().getTime());
				holidayMap.put(h, value);
				size--;
			}

			if (holidayMap.containsKey(s)) {
				String value = sdf.format(lauar.getCalendar().getTime());
				holidayMap.put(s, value);
				size--;
			}
			lauar.nextDay();
		}
	}

	// 节日转化为时间
	private TimeFormat holidayDate(String holiday) {
		String date = null;
		if (holidayMap.containsKey(holiday)) {
			date = holidayMap.get(holiday);
		} else if (!holidayMap.containsKey(holiday) && holiday.contains("节")) {
			holiday = holiday.replace("节", "");
			date = holidayMap.get(holiday);
		}

		TimeFormat tf = null;
		if (date != null)
			tf = new TimeFormat(date, date, null);
		// System.out.println(tf.toString());
		return tf;
	}

	// 文字转换为标准时间
	/**
	 * @param unit
	 *            text修饰的时间单位,如"最近几天",单位是"天".支持年月日时分.与Calendar类中的单位标识已知
	 * @param position
	 *            时间轴区间标识,0标识当前,1标识未来,-1标识过去
	 * @param movelength
	 *            偏移的长度
	 * @return
	 */
	private TimeFormat moveTime(int unit, int position, int movelength) {
		// 时间推移的长度
		TimeFormat tf = null;
		Calendar cal = Calendar.getInstance();
		String nowTime = sdf.format(cal.getTime());
		cal.add(unit, (movelength - 1) * position);
		String moveTime = sdf.format(cal.getTime());
		if (position == -1)
			tf = new TimeFormat(moveTime, nowTime, null);
		else
			tf = new TimeFormat(nowTime, moveTime, null);
		return tf;
	}

	// 主要函数
	public TimeFormat normTime(String text) {
		TimeFormat tf = null;
		try {
			tf = holidayDate(text);
			if (tf == null)
				tf = weekTime(text);
			if (tf == null)
				tf = day(text);
		} catch (Exception e) {
			e.getMessage();
		}
		return tf;
	}

	private TimeFormat day(String text) {
		TimeFormat tf = null;
		text = text.trim();
		String[] h = { "五一", "六一", "七一", "八一", "九一", "十一" };
		for (int i = 0; i < h.length; i++) {
			String e = h[i];
			if (text.equals(e)) {
				Calendar cal = Calendar.getInstance();
				cal.set(Calendar.MONTH, i + 4);
				cal.set(Calendar.DAY_OF_MONTH, 1);
				String moveTime = sdf.format(cal.getTime());
				tf = new TimeFormat(moveTime, moveTime, null);
			}
		}
		return tf;
	}

	private TimeFormat weekTime(String text) {
		TimeFormat tf = null;
		// 处理 感恩节、父亲节、母亲节等这类的节日
		if (text.trim().contains("节")) {
			String[] s = { "感恩节", "父亲节", "母亲节" };
			for (int j = 0; j < s.length; j++) {
				Calendar cal = Calendar.getInstance();
				if (text.equals(s[0])) {
					cal.set(Calendar.MONTH, 10);
					cal.set(Calendar.WEEK_OF_MONTH, 4);
					cal.set(Calendar.DAY_OF_WEEK, Calendar.THURSDAY);
				} else if (text.equals(s[1])) {
					cal.set(Calendar.MONTH, 5);
					cal.set(Calendar.WEEK_OF_MONTH, 3);
					cal.set(Calendar.DAY_OF_WEEK, Calendar.SUNDAY);
				} else if (text.equals(s[2])) {
					cal.set(Calendar.MONTH, 4);
					cal.set(Calendar.WEEK_OF_MONTH, 2);
					cal.set(Calendar.DAY_OF_WEEK, Calendar.SUNDAY);
				}
				String moveTime = sdf.format(cal.getTime());
				tf = new TimeFormat(moveTime, moveTime, null);

			}
		}
		return tf;
	}

	private HashMap<String, String> createMap() {
		HashMap<String, String> map = new HashMap<>();
		for (String e : holidays) {
			map.put(e, "");
		}
		return map;
	}

	public static void main(String[] args) {

		String[] riqiStrings = { "十一","父亲节", "元旦", "除夕", "春节", "清明节", "劳动节", "端午节", "中秋节", "国庆节", "母亲节", "儿童节", "建军节", "愚人节",
				"青年节", "圣诞节", "平安夜", "教师节", "万圣节", "植树节", "重阳节", "腊八节", "情人节", "元宵节", "感恩节", "妇女节", "小年" };

		StandardTime standardTime = new StandardTime();
		for (String str : riqiStrings) {
			System.out.println(str);
			TimeFormat tfFormat = standardTime.normTime(str);
			System.out.println(tfFormat.getStartTime());
			System.out.println(tfFormat.getEndTime());
		}
	}
}
