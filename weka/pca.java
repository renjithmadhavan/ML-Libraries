import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.attributeSelection.PrincipalComponents;
import weka.attributeSelection.Ranker;

public class pca {
	public static void main(String args[]) {
		try {
			// Load the Data.
			DataSource source = new DataSource("../data/ingredients.arff");
			Instances data = source.getDataSet();

			// Perform PCA.
			PrincipalComponents pca = new PrincipalComponents();
			pca.setVarianceCovered(1.0);
			pca.setCenterData(true);
			pca.setTransformBackToOriginal(false);
			pca.buildEvaluator(data);

			// Show transform data into eigenvector basis.
			Instances transformedData = pca.transformedData(data);
			System.out.println(transformedData);
		} catch (Exception e) {
			e.printStackTrace();
		}
	}

}
