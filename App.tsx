import React, { useState } from 'react';
import { View, Text, TextInput, Button, StyleSheet, Alert } from 'react-native';
import { TransformerOps } from './TransformerOps';

const App: React.FC = () => {
  const [text1, setText1] = useState<string>('');
  const [category, setCategory] = useState<string>(null);

  const calculateSimilarity = async () => {
    try {
      const transformerOps = await TransformerOps.getInstance();
      const output = await transformerOps.runInference(text1); // Ensure this is awaited
      setCategory(output);
    } catch (error) {
      console.error('Error calculating similarity:', error);
      Alert.alert('An error occurred while calculating similarity.');
    }
  };

  return (
    <View style={styles.container}>
      <TextInput
        style={styles.input}
        value={text1}
        onChangeText={setText1}
      />
      <Button title="Check" onPress={calculateSimilarity} />
      {category !== null && (
        <Text style={styles.similarityText}>
          Cluster: {category}
        </Text>
      )}
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    justifyContent: 'center',
    paddingHorizontal: 20,
    backgroundColor: '#f5f5f5',
  },
  title: {
    fontSize: 24,
    fontWeight: 'bold',
    marginBottom: 20,
    textAlign: 'center',
  },
  input: {
    borderWidth: 1,
    borderColor: '#ccc',
    padding: 10,
    marginBottom: 20,
    borderRadius: 5,
    backgroundColor: '#fff',
  },
  similarityText: {
    marginTop: 20,
    fontSize: 18,
    textAlign: 'center',
  },
});

export default App;
