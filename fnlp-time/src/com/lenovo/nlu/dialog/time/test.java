package com.lenovo.nlu.dialog.time;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.InputStreamReader;
import java.util.List;

public class test {
	
	
	
	

	public static void main(String[] args) {
		// TODO Auto-generated method stub
		
		String filepath="/home/gengsq2/java-neon/workspace/fnlp-time/src/com/lenovo/nlu/dialog/time/timetest.txt";
		String line = "";
		try {
			FileInputStream fis = new FileInputStream(filepath);// 创建对应f的文件输入流
			InputStreamReader isr = new InputStreamReader(fis, "UTF-8");
			BufferedReader br = new BufferedReader(isr);
			while ((line = br.readLine()) != null) {
					List<TimeFormat2> ltf = StdTime2.normalTime(line);
//					System.out.println("ltf.size():" + ltf.size());
					for (TimeFormat2 t : ltf) {
						System.out.println(t.toString());
					}
			}

		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}

	}

}
