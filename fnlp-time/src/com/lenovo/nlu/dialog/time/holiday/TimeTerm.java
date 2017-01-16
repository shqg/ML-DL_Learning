package com.lenovo.nlu.dialog.time.holiday;

import java.util.Arrays;

public class TimeTerm {
	public final static String[] HOLIDAY = { "小寒", "大寒", "立春", "雨水", "惊蛰",
			"春分", "清明", "谷雨", "立夏", "小满", "芒种", "夏至", "小暑", "大暑", "立秋", "处暑",
			"白露", "秋分", "寒露", "霜降", "立冬", "小雪", "大雪", "冬至", "春节", "元宵", "端午",
			"七夕", "中元", "中秋", "重阳", "腊八", "小年", "除夕", "元旦", "情人节", "妇女节",
			"植树节", "消费者权益日", "愚人节", "劳动节", "青年节", "护士节", "儿童节", "建党节", "建军节",
			"爸爸节", "教师节", "孔子诞辰", "国庆节", "老人节", "联合国日", "孙中山诞辰纪念", "澳门回归纪念",
			"平安夜","万圣节","圣诞"};
	public final static String[] NEAR ={"最近","附近"};
	public final static String[] START = {"初"};
	public final static String[] MIDDLE = {"中"};
	public final static String[] END = {"底","末","下旬"};
	public final static String[] NOW = {"现在","当下","当前","今","本"};
	public final static String[] LAST = {"过去","前"};
	public final static String[] UNIT = {"年","月","周","日","时","分","秒"};
	public final static String[] POPULAR ={"五一","十一","六一","五四"};
	
	static{
		Arrays.sort(HOLIDAY);
	}
}

