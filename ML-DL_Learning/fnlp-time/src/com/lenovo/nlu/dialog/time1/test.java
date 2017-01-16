package com.lenovo.nlu.dialog.time1;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.InputStreamReader;
import java.util.List;

public class test {

	public static void main(String[] args) {
		// TODO Auto-generated method stub

		String filepath = "/home/gengsq2/java-neon/workspace2/fnlp-time/src/com/lenovo/nlu/dialog/time1/timetest.txt";
		String line = "";
		try {
			FileInputStream fis = new FileInputStream(filepath);// 创建对应f的文件输入流
			InputStreamReader isr = new InputStreamReader(fis, "UTF-8");
			BufferedReader br = new BufferedReader(isr);
			while ((line = br.readLine()) != null) {
				List<TimeFormat2> ltf = StdTime2.normalTime(line);
				List<TimeFormat2> ltf2 = StdTime2.normalTime_o(line, ltf);
				// System.out.println("ltf.size():" + ltf.size());
				for (TimeFormat2 t : ltf2) {
					System.out.println(t.toString());
				}
				List<TimeFormat3> ltf3=StdTime2.normalTime_o2(line,ltf2);
				System.out.println("ltf3.size():" + ltf3.size());
				for (TimeFormat3 t : ltf3) {
					System.out.println(t.toString());
				}
			}

		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}

	}

}
